# ImageForge v0.7 "Reference-Lock"
# Bulk blog images with CC/non-CC thumbnail referencing, hard constraints, venue mode,
# fidelity slider, and auto-retry guards. Pinterest/long-pin ready (toggle in Advanced).

import os
import io
import re
import math
import time
import base64
import json
import zipfile
import random
from typing import List, Dict, Optional, Tuple

import requests
from PIL import Image, ImageStat
import streamlit as st

# -----------------------------
# ---------- CONFIG -----------
# -----------------------------

APP_TITLE = "ImageForge v0.7 — Reference-Lock"
DEFAULT_OUTPUT_W, DEFAULT_OUTPUT_H = 1200, 675          # blog hero 16:9
PIN_OUTPUT_W, PIN_OUTPUT_H = 1000, 1500                 # Pinterest standard (optional)
OPENAI_SIZES = ["1536x1024", "1024x1536", "1024x1024"]  # model-allowed base renders

SITE_PROFILES = {
    "vailvacay.com":  "Colorado Rockies resort/village, alpine architecture, fir/spruce, riverside paths, ski terrain; photoreal; editorial stock feel; no text/logos.",
    "bostonvacay.com":"Historic New England city, brick & brownstone, harbor/waterfront, parks; photoreal; editorial stock feel; no text/logos.",
    "bangkokvacay.com":"Southeast Asian metropolis, street food/night markets, temples/rooftops/skyline; photoreal; editorial stock feel; no text/logos.",
    "ipetzo.com":      "Pet lifestyle scenes, owners with dogs/cats in parks/neutral interiors; photoreal; editorial stock feel; no brands/text.",
    "1-800deals.com":  "Generic shopping/ecommerce scenes, parcels, aisles; clean backgrounds; photoreal; no brands/text."
}
DEFAULT_SITE = "vailvacay.com"

# -----------------------------
# --------- UTILITIES ---------
# -----------------------------

def slugify(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[’'`]", "", t)
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t or "image"

def crop_to_aspect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    target_ratio = target_w / target_h
    w, h = img.size
    cur_ratio = w / h
    if cur_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        box = (left, 0, left + new_w, h)
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        box = (0, top, w, top + new_h)
    return img.crop(box)

def save_webp_bytes(img: Image.Image, w: int, h: int, quality: int) -> bytes:
    img = img.convert("RGB")
    img = crop_to_aspect(img, w, h).resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=quality, method=6)
    return buf.getvalue()

def fetch_image_bytes(url: str, timeout: int = 20) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        pass
    return None

def luminance_fraction(img: Image.Image, lo: int = 220) -> float:
    # frac of near-white pixels (quick snow heuristic)
    g = img.convert("L")
    hist = g.histogram()
    total = sum(hist)
    white = sum(hist[lo:])
    return (white / total) if total else 0.0

def color_presence(img: Image.Image, hue_target: Tuple[int,int], sat_lo=80, val_lo=70) -> float:
    # simple % of pixels in hue window (0-360), HSV space; coarse “yellow umbrellas”/“blue sky”
    img = img.convert("RGB").resize((256, 256))
    import colorsys
    w, h = img.size
    count, hit = 0, 0
    h1, h2 = hue_target
    wrap = h2 < h1
    for y in range(h):
        for x in range(w):
            r,g,b = img.getpixel((x,y))
            r/=255; g/=255; b/=255
            H,S,V = colorsys.rgb_to_hsv(r,g,b)
            Hdeg, S100, V100 = int(H*360), int(S*100), int(V*100)
            count += 1
            cond_h = (Hdeg>=h1 or Hdeg<=h2) if wrap else (h1<=Hdeg<=h2)
            if cond_h and S100>=sat_lo and V100>=val_lo:
                hit += 1
    return hit/max(1,count)

def analyze_reference(url: str) -> Dict:
    """Return coarse features extracted from a reference image URL (never sent to OpenAI)."""
    data = fetch_image_bytes(url)
    if not data:
        return {"ok": False}
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return {"ok": False}

    # Heuristics
    snow_frac = luminance_fraction(img, lo=225)           # near-white area
    yellow_frac = color_presence(img, (35, 60), 70, 65)   # yellow umbrellas/lights
    blue_frac = color_presence(img, (195, 240), 40, 55)   # sky-ish

    # Palettes & warmth
    stat = ImageStat.Stat(img)
    mean = stat.mean  # simple palette hint
    warm_hint = (mean[0] > mean[2])  # R > B => slightly warm

    # Very coarse checklist guess
    checklist = set()
    if snow_frac >= 0.06:       # ~6%+
        checklist.add("snow_on_ground_and_roofs")
    if yellow_frac >= 0.015:
        checklist.add("patio_umbrellas_or_warm_yellow_accents")
    # Assume alpine façade if lots of verticals/warm timber-like palette: always include; safe generic
    checklist.add("alpine_gabled_facade_with_balconies")
    if blue_frac >= 0.04:
        checklist.add("visible_sky_background")

    return {
        "ok": True,
        "snow_frac": snow_frac,
        "yellow_frac": yellow_frac,
        "blue_frac": blue_frac,
        "warm_hint": warm_hint,
        "checklist": list(checklist)
    }

def expectation_from_features(feat: Dict) -> Dict:
    return {
        "expect_snow": "snow_on_ground_and_roofs" in feat.get("checklist", []),
        "expect_yellow": "patio_umbrellas_or_warm_yellow_accents" in feat.get("checklist", [])
    }

def postcheck(img_bytes: bytes, expect: Dict) -> bool:
    """Verify key expectations (snow/yellow) appear; if missing, signal retry."""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return False
    ok = True
    if expect.get("expect_snow"):
        if luminance_fraction(img, lo=225) < 0.04:
            ok = False
    if expect.get("expect_yellow"):
        if color_presence(img, (35, 60), 70, 65) < 0.008:
            ok = False
    return ok

# -----------------------------
# ---- SEARCH THUMBNAILS ------
# -----------------------------

def openverse_search(q: str, n: int = 4) -> List[Dict]:
    # Public CC search (no key required for basic use)
    # Using /v1/images? q & license= (broad). We only need a few thumbs.
    url = "https://api.openverse.engineering/v1/images/"
    params = {"q": q, "license": "cc0,cc-by,cc-by-sa,cc-by-nd", "page_size": min(10, max(1,n))}
    try:
        r = requests.get(url, params=params, timeout=20)
        items = []
        if r.status_code == 200:
            data = r.json()
            for it in data.get("results", []):
                link = it.get("url") or it.get("foreign_landing_url")
                thumb = it.get("thumbnail") or link
                if link:
                    items.append({
                        "title": it.get("title") or "",
                        "url": link,
                        "thumb": thumb,
                        "source": "Openverse (CC)",
                        "license": it.get("license") or "cc",
                        "is_cc": True
                    })
                if len(items) >= n:
                    break
        return items
    except Exception:
        return []

def google_cse_images(q: str, api_key: str, cx: str, n: int = 4) -> List[Dict]:
    if not api_key or not cx:
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key, "cx": cx, "q": q, "searchType": "image",
        "num": min(10, max(1, n)), "safe": "active", "imgSize": "large"
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        out = []
        if r.status_code == 200:
            data = r.json()
            for it in data.get("items", []):
                link = it.get("link")
                thumb = it.get("image", {}).get("thumbnailLink") or link
                if link:
                    out.append({
                        "title": it.get("title") or "",
                        "url": link,
                        "thumb": thumb,
                        "source": "Google (non-CC)",
                        "license": "reference-only",
                        "is_cc": False
                    })
                if len(out) >= n:
                    break
        return out
    except Exception:
        return []

# -----------------------------
# ---------- PROMPTS ----------
# -----------------------------

def venue_mode_hint(keyword: str) -> bool:
    k = keyword.lower()
    return any(x in k for x in [
        "restaurant","bar","cafe","coffee","hotel","inn","lodge","tavern","rooftop","resort"
    ])

def detect_season_from_keyword(keyword: str) -> Optional[str]:
    k = keyword.lower()
    for name, tag in [
        ("january","winter"),("february","winter"),("march","late winter"),
        ("april","spring shoulder"),("may","spring"),
        ("june","summer"),("july","summer"),("august","summer"),
        ("september","early fall"),("october","fall"),("november","late fall"),
        ("december","winter"),
        ("christmas","winter"),("snow","winter"),("ski","winter"),("back bowls","winter")
    ]:
        if name in k:
            return tag
    return None

def build_prompt(site: str,
                 keyword: str,
                 season_mode: str,
                 ref_feat: Optional[Dict],
                 fidelity: int,
                 lock_strength: str,
                 venue: bool) -> str:

    base = SITE_PROFILES.get(site, SITE_PROFILES[DEFAULT_SITE])
    season_hint = ""
    if season_mode == "Auto":
        s = detect_season_from_keyword(keyword) or ""
        season_hint = f"Season cues: {s}. " if s else ""
    elif season_mode in ("Winter","Summer","Fall","Spring"):
        season_hint = f"Season cues: {season_mode}. "

    # Reference checklist -> constraints
    constraints = []
    if ref_feat and ref_feat.get("ok"):
        cl = set(ref_feat.get("checklist", []))
        names = {
            "alpine_gabled_facade_with_balconies": "alpine gabled façade with balconies",
            "snow_on_ground_and_roofs": "fresh snow on ground and roofs",
            "patio_umbrellas_or_warm_yellow_accents": "patio with umbrellas and warm yellow accents",
            "visible_sky_background": "sky and mountain backdrop"
        }
        for key in ["alpine_gabled_facade_with_balconies",
                    "snow_on_ground_and_roofs",
                    "patio_umbrellas_or_warm_yellow_accents",
                    "visible_sky_background"]:
            if key in cl:
                constraints.append(names[key])

    # Lock strength → language
    must = ("must clearly include" if fidelity >= 60 else "should include")
    lock_phrase = ""
    if lock_strength == "Strong" and constraints:
        lock_phrase = f" The scene {must}: " + ", ".join(constraints) + "."
    elif lock_strength == "Moderate" and constraints:
        lock_phrase = f" The scene should feature: " + ", ".join(constraints) + "."

    # Venue mode extras
    venue_bits = ""
    if venue:
        venue_bits = (" Exterior, eye-level view of the venue façade and patio seating if present; "
                      "warm window glow or string lights at dusk is welcome; "
                      "do not show readable signage or logos; keep everything generic.")

    # Fidelity → strictness keywords
    strict = ""
    if fidelity >= 80:
        strict = (" Balanced yet tight composition mirroring the reference layout. "
                  "Prominent, unambiguous depiction of the listed elements.")
    elif fidelity >= 50:
        strict = " Keep composition similar to the reference vibe and include the listed elements."
    else:
        strict = " General vibe match is sufficient."

    prompt = (
        f"{base} {season_hint}"
        f"Create a photorealistic, editorial-style stock image for: '{keyword}'. "
        f"Landscape orientation. No words, watermarks, or trademarks; no readable storefront text. "
        f"{venue_bits}{lock_phrase}{strict}"
    )
    return prompt

# -----------------------------
# --------- OPENAI IMG --------
# -----------------------------

def dalle_generate_png_bytes(prompt: str, size: str, api_key: str) -> bytes:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-image-1",
        "prompt": prompt,
        "size": size,
        "response_format": "b64_json"
    }
    r = requests.post("https://api.openai.com/v1/images/generations",
                      headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
    b64 = r.json()["data"][0]["b64_json"]
    return base64.b64decode(b64)  # PNG bytes

# -----------------------------
# ------------- UI ------------
# -----------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Keys")
    openai_key = st.text_input("OpenAI API key", type="password")
    g_key = st.text_input("Google CSE API key (for thumbnails)", type="password")
    g_cx  = st.text_input("Google CSE CX (engine id)", type="password")

    st.subheader("Output")
    site = st.selectbox("Site style", list(SITE_PROFILES.keys()),
                        index=list(SITE_PROFILES.keys()).index(DEFAULT_SITE))
    render_size = st.selectbox("Render base size", OPENAI_SIZES, index=0)
    webp_quality = st.slider("WebP quality", 60, 95, 82)
    make_pins = st.checkbox("Also make a Pinterest image (1000×1500)", value=False)

    st.subheader("Reference control")
    prefer_cc = st.checkbox("Try Creative-Commons photo first", value=True)
    show_non_cc = st.checkbox("If no CC found, allow non-CC thumbnails for reference", value=True)
    pick_mode = st.selectbox("Thumbnail picking", ["Auto pick best", "Manual pick (thumbnails)"])
    cand_per_kw = st.slider("Candidates per keyword", 2, 8, 4)

    st.subheader("Lock & season")
    lock_strength = st.selectbox("Reference Lock", ["Strong", "Moderate", "Off"], index=0)
    fidelity = st.slider("Fidelity (strictness & retries)", 0, 100, 70)
    season_mode = st.selectbox("Season cues", ["Auto", "Winter", "Summer", "Fall", "Spring"])

    st.subheader("Advanced")
    season_aware = st.checkbox("Season-aware prompts", value=True)
    images_per_kw = st.slider("Images per keyword", 1, 5, 1)

st.caption("Paste keywords (one per line).")
keywords_text = st.text_area("Keywords", height=180,
    placeholder="Tavern on the Square, Vail Colorado\nBest seafood restaurant in Boston\nThings to do in Vail in October")

colA, colB = st.columns([1,1])
generate_btn = colA.button("Generate", type="primary")
clear_btn = colB.button("Clear")
if clear_btn:
    st.session_state.clear()
    st.experimental_rerun()

def find_thumbnails(q: str) -> List[Dict]:
    items: List[Dict] = []
    if prefer_cc:
        items.extend(openverse_search(q, n=cand_per_kw))
    if (not items) and show_non_cc:
        items.extend(google_cse_images(q, g_key, g_cx, n=cand_per_kw))
    return items[:cand_per_kw]

def show_thumb_picker(q: str, items: List[Dict]) -> Optional[Dict]:
    if not items:
        st.info("No CC candidates found. It will use AI (with reference lock off) for this one.")
        return None
    pick = None
    for i, it in enumerate(items):
        with st.container():
            cols = st.columns([3,5])
            with cols[0]:
                st.image(it.get("thumb") or it["url"], use_container_width=True)
            with cols[1]:
                st.write(f"**{it.get('title') or 'Untitled'}**")
                st.caption(f"Source: {it['source']}, license: {it['license']}")
                if st.button(f"Use this as reference (#{i+1})", key=f"use_{slugify(q)}_{i}"):
                    pick = it
    return pick

def autopick_thumb(items: List[Dict]) -> Optional[Dict]:
    if not items:
        return None
    # Prefer CC first; otherwise first non-CC
    for it in items:
        if it.get("is_cc"):
            return it
    return items[0]

if generate_btn:
    if not openai_key:
        st.warning("Please enter your OpenAI API key.")
        st.stop()
    if not keywords_text.strip():
        st.warning("Please paste at least one keyword.")
        st.stop()

    keywords = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
    # Output buffers
    zip_buf = io.BytesIO()
    zf = zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED)

    progress = st.progress(0)
    status = st.empty()
    gallery = []

    for idx, kw in enumerate(keywords, start=1):
        status.info(f"Working {idx}/{len(keywords)}: {kw}")
        venue = venue_mode_hint(kw)

        # ---- Reference discovery / selection
        picked: Optional[Dict] = None
        thumb_items = find_thumbnails(kw)
        with st.expander(f"📷 CC Thumbnails — choose one or leave 'AI render': {kw}", expanded=False):
            if pick_mode.startswith("Manual"):
                picked = show_thumb_picker(kw, thumb_items)
            else:
                picked = autopick_thumb(thumb_items)

        # Analyze reference locally (never uploaded)
        ref_feat = None
        expect = {}
        if picked:
            st.caption(f"Using reference: {picked.get('title') or picked['source']} ({picked['source']})")
            feat = analyze_reference(picked["url"])
            if feat.get("ok"):
                ref_feat = feat
                expect = expectation_from_features(feat)

        # ---- Build prompt
        lock = lock_strength if ref_feat else "Off"
        if not season_aware:
            season_mode_eff = "Auto"
        else:
            season_mode_eff = season_mode

        prompt = build_prompt(site, kw, season_mode_eff, ref_feat, fidelity, lock, venue)

        # ---- Render N images with auto-retry
        num_needed = images_per_kw
        made = 0
        tries_per_img = 1 + (fidelity // 40)  # 1..3 tries/img
        while made < num_needed:
            ok_img: Optional[bytes] = None
            local_prompt = prompt
            for attempt in range(tries_per_img):
                try:
                    png = dalle_generate_png_bytes(local_prompt, render_size, openai_key)
                except Exception as e:
                    st.error(f"{kw}: OpenAI error — {e}")
                    break
                # Check expectations
                if ref_feat and lock != "Off":
                    if postcheck(png, expect):
                        ok_img = png
                        break
                    else:
                        # strengthen wording
                        local_prompt += " The required elements must be clearly visible in the final image."
                else:
                    ok_img = png
                    break
            if not ok_img:
                # give up on constraints, keep the latest (so user gets something)
                ok_img = png

            # Save blog webp
            try:
                im = Image.open(io.BytesIO(ok_img)).convert("RGB")
            except Exception:
                st.error(f"{kw}: Failed to decode image bytes.")
                break

            base_slug = slugify(kw)
            fname_blog = f"{base_slug}.webp" if images_per_kw == 1 else f"{base_slug}-{made+1}.webp"
            webp_bytes = save_webp_bytes(im, DEFAULT_OUTPUT_W, DEFAULT_OUTPUT_H, webp_quality)
            zf.writestr(fname_blog, webp_bytes)
            gallery.append((fname_blog, webp_bytes))

            # Optional Pinterest
            if make_pins:
                fname_pin = f"{base_slug}-pin-{made+1 if images_per_kw>1 else '1'}.webp"
                pin_bytes = save_webp_bytes(im, PIN_OUTPUT_W, PIN_OUTPUT_H, webp_quality)
                zf.writestr(fname_pin, pin_bytes)
                gallery.append((fname_pin, pin_bytes))

            made += 1

        progress.progress(idx/len(keywords))

    zf.close()
    zip_buf.seek(0)
    st.success("Done! Download your images below.")
    st.download_button("⬇️ Download ZIP", data=zip_buf,
                       file_name=f"{slugify(site)}_images.zip", mime="application/zip")

    st.markdown("### Previews & individual downloads")
    cols = st.columns(3)
    for i, (fname, data_bytes) in enumerate(gallery):
        with cols[i % 3]:
            st.image(data_bytes, caption=fname, use_container_width=True)
            st.download_button("Download", data=data_bytes, file_name=fname, mime="image/webp", key=f"btn_{i}")
