# ImageForge v0.9.4 (CC-SmartFallback)
# CC-first thumbnails with safe non-CC "reference only" fallback + AI rendering.
# Requirements: streamlit, requests, pillow

import os, io, re, json, time, math, base64, zipfile, datetime
from typing import List, Dict, Optional, Tuple
import requests
from PIL import Image
import streamlit as st

# ---------------------------- Config ---------------------------- #

APP_TITLE = "ImageForge – CC-SmartFallback"
DEFAULT_OUTPUT_W, DEFAULT_OUTPUT_H = 1200, 675
DEFAULT_WEBP_QUALITY = 82
OPENAI_IMG_SIZES = ["1536x1024", "1024x1536", "1024x1024"]  # cropped to target later

SITE_PROFILES = {
    "vailvacay.com":  "Photorealistic alpine resort & village scenes in the Colorado Rockies; gondolas, riverwalks, cozy lodges; no text.",
    "bangkokvacay.com":"Photorealistic Bangkok city life; street food, temples, river views, neon night scenes; no brands; no text.",
    "bostonvacay.com": "Photorealistic New England/Boston; brownstones, harbor, historic streets; no brands; no text.",
    "ipetzo.com":      "Photorealistic pet lifestyle; dogs/cats with owners outdoors/indoors; neutral interiors/parks; no visible logos or text.",
    "1-800deals.com":  "Photorealistic retail/ecommerce visuals; shopping scenes, parcels; generic products; clean backgrounds; no brands or text.",
}

# ------------------------- Utilities ---------------------------- #

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

def to_webp_bytes(img: Image.Image, w: int, h: int, quality: int) -> bytes:
    img = img.convert("RGB")
    img = crop_to_aspect(img, w, h).resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=quality, method=6)
    return buf.getvalue()

def season_hint(keyword: str, site: str) -> str:
    k = keyword.lower()
    # Minimal hints; you can expand later
    if "winter" in k or "december" in k or "january" in k or "february" in k:
        return "winter conditions, snow present"
    if "summer" in k or "july" in k or "august" in k or "june" in k:
        return "summer conditions, lush green"
    if "fall" in k or "autumn" in k or "september" in k or "october" in k or "november" in k:
        return "autumn foliage"
    if "spring" in k or "april" in k or "may" in k or "march" in k:
        # Vail nuance: spring may still have snow on mountain
        if "vail" in site or "vail" in k:
            return "late winter/early spring in the mountains; snow remains on ski runs"
        return "springtime"
    return "seasonally appropriate conditions"

def build_prompt(site: str, keyword: str, season_on: bool) -> str:
    base = SITE_PROFILES.get(site, SITE_PROFILES["vailvacay.com"])
    style_hints = []

    k = keyword.lower()

    if any(x in k for x in ["how far","directions","from","between"]):
        style_hints.append("scenic travel approach or roadway vantage; safe roadside view")
    if any(x in k for x in ["back bowls","ski","lift","gondola"]):
        style_hints.append("wide alpine vista that matches topic")
    if any(x in k for x in ["hotel","where to stay","resort"]):
        style_hints.append("inviting lodging exterior/interior, warm light")
    if "cigar" in k:
        style_hints.append("cozy lounge interior, warm ambient light, no branding")
    if any(x in k for x in ["don't ski","non-skiers","baby","kid","with kids","indoor"]):
        style_hints.append("family-friendly vibe, gentle activities")
    if any(x in k for x in ["county","river","mountain range","history","when did","when was"]):
        style_hints.append("editorial/documentary feel emphasizing place over people")

    season = season_hint(keyword, site) if season_on else "seasonally appropriate"
    if "ski patrol" in k and "snow" not in k:
        style_hints.append("active ski patrol with snow present on slopes (avoid green summer grass)")

    style = ", ".join([s for s in style_hints if s]) or "scene appropriate to the topic"

    return (f"{base} Balanced composition; natural light; editorial stock-photo feel. "
            f"Create an image for the topic: '{keyword}'. Landscape orientation. No words or typography. "
            f"Ensure {season}. Scene intent: {style}.")

# ---------------------- OpenAI Image Gen ------------------------ #

def openai_generate_image(prompt: str, size: str, api_key: str) -> Image.Image:
    """
    Robust against API variations:
    - Tries images/generations with default response (URL).
    - If b64_json present, decodes it.
    """
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-image-1",
        "prompt": prompt,
        "size": size
        # Do NOT send response_format to avoid 400 "unknown parameter"
    }
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")

    data0 = r.json().get("data", [{}])[0]
    img_bytes = None

    if "b64_json" in data0:
        img_bytes = base64.b64decode(data0["b64_json"])
    elif "url" in data0 and data0["url"]:
        getr = requests.get(data0["url"], timeout=180)
        getr.raise_for_status()
        img_bytes = getr.content
    else:
        raise RuntimeError("OpenAI returned no image data.")

    return Image.open(io.BytesIO(img_bytes))

# ---------------------- CC Search (Openverse) ------------------- #

def openverse_search(query: str, num: int = 4) -> List[Dict]:
    """
    Commercial-friendly CC images via Openverse.
    Returns list of dicts with keys: title, thumb_url, full_url, source, license
    """
    try:
        params = {
            "q": query,
            "license_type": "commercial",  # only commercial-friendly licenses
            "page": 1,
            "page_size": max(1, min(num, 20))
        }
        r = requests.get("https://api.openverse.engineering/v1/images/", params=params, timeout=30)
        if r.status_code != 200:
            return []
        results = []
        for item in r.json().get("results", []):
            full = item.get("url") or item.get("foreign_landing_url")
            thumb = item.get("thumbnail") or full
            if not full:
                continue
            results.append({
                "title": item.get("title") or "Openverse image",
                "thumb_url": thumb,
                "full_url": full,
                "source": "Openverse (CC)",
                "license": item.get("license") or "cc",
                "provider": item.get("provider")
            })
        return results
    except Exception:
        return []

# ------------------ Reference Search (SerpAPI) ------------------ #

def serpapi_google_images(query: str, serp_key: str, num: int = 4) -> List[Dict]:
    """
    Non-CC thumbnails strictly for reference/inspiration (NOT saved).
    """
    if not serp_key:
        return []
    try:
        params = {
            "engine": "google_images",
            "q": query,
            "ijn": "0",
            "api_key": serp_key
        }
        r = requests.get("https://serpapi.com/search", params=params, timeout=40)
        if r.status_code != 200:
            return []
        imgs = r.json().get("images_results", []) or []
        out = []
        for it in imgs[:num]:
            thumb = it.get("thumbnail") or it.get("original")
            full = it.get("original") or it.get("thumbnail")
            if not full:
                # Defensive: skip items without fetchable url
                continue
            out.append({
                "title": it.get("title") or "Google image",
                "thumb_url": thumb,
                "full_url": full,
                "source": "Google (non-CC)",
                "license": "Reference only (non-CC)"
            })
        return out
    except Exception:
        return []

# ----------------- Candidate Aggregation & Save ----------------- #

def build_cc_query(keyword: str) -> str:
    """Broaden venue queries a bit; never append site name."""
    q = keyword.strip()
    if "vail" not in q.lower() and "bangkok" not in q.lower() and "boston" not in q.lower():
        # nudge geolocation if the keyword lacks place context
        q += " exterior storefront"
    # For Vail venue names, make it clearer
    if "vail" in q.lower() and not any(w in q.lower() for w in ["exterior","storefront","restaurant","hotel","lodge"]):
        q += " exterior storefront"
    return q

def aggregate_candidates(keyword: str,
                         want_sources: List[str],
                         serp_key: str,
                         num: int,
                         prefer_cc: bool,
                         allow_ref_noncc: bool) -> List[Dict]:
    """
    Returns candidate dicts with keys: title, thumb_url, full_url, source, license
    """
    q = build_cc_query(keyword)
    results: List[Dict] = []

    # Always try Openverse first when CC is preferred
    if prefer_cc and "Openverse" in want_sources:
        results.extend(openverse_search(q, num=num))

    # If still empty and allowed, show reference-only non-CC via Google
    if not results and allow_ref_noncc and "Google (via SerpAPI)" in want_sources:
        results.extend(serpapi_google_images(q, serp_key, num=num))

    # If CC isn't preferred at all, we can still show reference thumbnails (not saved)
    if not prefer_cc and "Google (via SerpAPI)" in want_sources and not results:
        results.extend(serpapi_google_images(q, serp_key, num=num))

    # De-dupe by full_url
    seen = set()
    deduped = []
    for r in results:
        fu = r.get("full_url")
        if not fu or fu in seen:
            continue
        seen.add(fu)
        deduped.append(r)

    return deduped

def download_image(url: str) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=40)
        if r.status_code != 200:
            return None
        return Image.open(io.BytesIO(r.content))
    except Exception:
        return None

def save_cc_candidate(cand: Dict, fname_base: str, out_w: int, out_h: int, q: int) -> Optional[bytes]:
    # Never save non-CC/reference items
    if "non-cc" in cand.get("source","").lower() or "reference" in cand.get("license","").lower():
        return None
    full = cand.get("full_url")
    if not full:
        return None
    img = download_image(full)
    if not img:
        return None
    return to_webp_bytes(img, out_w, out_h, q)

# --------------------------- UI -------------------------------- #

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("CC-first with safe reference thumbnails. Then fallback to OpenAI render, cropped to 1200×675.")

with st.sidebar:
    site = st.selectbox("Site style", list(SITE_PROFILES.keys()), index=0)
    out_w = st.number_input("Output width", 600, 2400, DEFAULT_OUTPUT_W, step=50)
    out_h = st.number_input("Output height", 300, 1350, DEFAULT_OUTPUT_H, step=25)
    webp_q = st.slider("WebP quality", 60, 95, DEFAULT_WEBP_QUALITY)

    st.markdown("### Generation")
    size = st.selectbox("OpenAI render size", OPENAI_IMG_SIZES, index=0)
    season_on = st.checkbox("Season-aware prompts", value=True)

    st.markdown("### API Keys")
    openai_key = st.text_input("OpenAI API key", type="password")
    serp_key = st.text_input("SerpAPI key (for Google thumbnails)", type="password")

    st.markdown("### Prefer Real Photos?")
    prefer_cc = st.checkbox("Try Creative-Commons photo first", value=True)
    allow_ref_noncc = st.checkbox("If no CC found, show non-CC thumbnails for reference only", value=True, disabled=not prefer_cc)

    st.markdown("### CC selection")
    cc_mode = st.selectbox("Thumbnail picking", ["Manual pick (thumbnails)"])

    st.markdown("### Sources to search")
    sources = st.multiselect("Sources", ["Google (via SerpAPI)", "Openverse"], default=["Google (via SerpAPI)", "Openverse"])

    cand_per_kw = st.slider("Candidates per keyword", 1, 8, 4)

st.markdown("#### Keywords (one per line)")
kw_text = st.text_area("", height=150, placeholder="Tavern on the Square, Vail Colorado\nBlue Moose Pizza in Vail Colorado\nThings to do in Vail in October")

cols = st.columns([1,1])
do_gen = cols[0].button("Generate")
if cols[1].button("Clear"):
    st.session_state.clear()
    st.experimental_rerun()

if do_gen:
    keywords = [ln.strip() for ln in kw_text.splitlines() if ln.strip()]
    if not keywords:
        st.warning("Please paste at least one keyword.")
        st.stop()
    if not openai_key:
        st.info("OpenAI key not set — you can still pick a CC photo (if available). AI fallback needs the key.")

    zip_buf = io.BytesIO()
    zf = zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED)

    previews: List[Tuple[str, bytes]] = []

    for idx, kw in enumerate(keywords, start=1):
        st.info(f"Working {idx}/{len(keywords)}: {kw}")
        exp = st.expander(f"📷  CC Thumbnails — choose one or leave 'AI render' · {kw}", expanded=True)

        # 1) Gather candidates
        cands = aggregate_candidates(
            keyword=kw,
            want_sources=sources,
            serp_key=serp_key,
            num=cand_per_kw,
            prefer_cc=prefer_cc,
            allow_ref_noncc=allow_ref_noncc
        )

        chosen_bytes: Optional[bytes] = None
        chosen_label: Optional[str] = None

        with exp:
            picked_idx = None
            if cands:
                thumbs = []
                options = []
                for i, c in enumerate(cands):
                    thumb = c.get("thumb_url") or c.get("full_url")
                    if not thumb:
                        continue
                    col1, col2 = st.columns([1,2])
                    with col1:
                        st.image(thumb, caption=f"{c.get('source','')}", use_container_width=True)
                    with col2:
                        st.write(f"**{c.get('title','Image')}**")
                        st.write(f"License: {c.get('license','n/a')}")
                        st.write(f"Source: {c.get('source','')}")
                    st.markdown("---")
                    thumbs.append(c)
                    options.append(f"{i+1}. {c.get('title','Image')} ({c.get('source','')})")

                if thumbs:
                    picked = st.radio("Pick a CC thumbnail (non-CC items are for reference only and will not be saved). "
                                      "Or leave unselected to use AI render.",
                                      ["Use AI render"] + options, index=0)
                    if picked != "Use AI render":
                        try:
                            picked_idx = int(picked.split(".")[0]) - 1
                        except Exception:
                            picked_idx = None

                # If user picked a CC candidate, try to save it now
                if picked_idx is not None and 0 <= picked_idx < len(thumbs):
                    cand = thumbs[picked_idx]
                    webp_bytes = save_cc_candidate(cand, slugify(kw), out_w, out_h, webp_q)
                    if webp_bytes:
                        fname = f"{slugify(kw)}.webp"
                        zf.writestr(fname, webp_bytes)
                        previews.append((fname, webp_bytes))
                        chosen_bytes = webp_bytes
                        chosen_label = "CC photo"
                        st.success("Saved Creative-Commons image.")
                    else:
                        st.warning("That candidate is non-CC or failed to download. Falling back to AI.")

            # 2) If nothing chosen/saved, do AI render
            if chosen_bytes is None:
                if not openai_key:
                    st.error("OpenAI key missing and no CC image saved. Skipping this keyword.")
                else:
                    try:
                        prompt = build_prompt(site, kw, season_on)
                        ai_img = openai_generate_image(prompt, size, openai_key)
                        webp_bytes = to_webp_bytes(ai_img, out_w, out_h, webp_q)
                        fname = f"{slugify(kw)}.webp"
                        zf.writestr(fname, webp_bytes)
                        previews.append((fname, webp_bytes))
                        chosen_bytes = webp_bytes
                        chosen_label = "AI render"
                        st.success("AI render saved.")
                    except Exception as e:
                        st.error(f"OpenAI generation failed: {e}")

        st.markdown("---")

    zf.close()
    zip_buf.seek(0)

    if previews:
        st.success("Done! Download your images below.")
        st.download_button("⬇️ Download ZIP", data=zip_buf,
                           file_name=f"{slugify(site)}_images.zip", mime="application/zip")

        st.markdown("### Previews & individual downloads")
        cols = st.columns(3)
        for i, (fname, data_bytes) in enumerate(previews):
            with cols[i % 3]:
                st.image(data_bytes, caption=fname, use_container_width=True)
                st.download_button("Download", data=data_bytes, file_name=fname, mime="image/webp")
    else:
        st.warning("Nothing was generated. Try different keywords or check API keys.")
