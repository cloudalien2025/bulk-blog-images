# ImageForge v1.4 — Real Photos + AI Render + Batch from Excel
# ------------------------------------------------------------
# Requirements: streamlit, requests, pillow, pandas, openpyxl
#   pip install streamlit requests pillow pandas openpyxl
#
# IMPORTANT: Enable these Google APIs if using Real Photos mode:
#   • Places API (or Places API (New))
#   • Street View Static API

from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
from PIL import Image, ImageOps
import streamlit as st
import pandas as pd

# -----------------------------
# App constants & helpers
# -----------------------------

APP_NAME = "ImageForge v1.4"
OUTPUT_W, OUTPUT_H = 1200, 675
PINTEREST_W, PINTEREST_H = 1000, 1500
OPENAI_IMAGE_SIZES = ("1536x1024", "1024x1536", "1024x1024")

SITE_PROFILES = {
    "vailvacay.com":  "Photoreal mountain resort scenes (Vail/Beaver Creek); alpine village; gondola lifts; Rocky Mountains; no logos; no text.",
    "bangkokvacay.com":"Photoreal Bangkok city life; night markets, Chinatown, BTS/MRT scenes; temples; tuk-tuks; warm ambient light; no logos; no text.",
    "bostonvacay.com": "Photoreal Boston & New England; brownstones, harbor, fall color; cafes; cobblestones; no logos; no text.",
    "ipetzo.com":      "Photoreal pet lifestyle; dogs and cats with people in tasteful settings; clean backgrounds; no visible branding; no text.",
    "1-800deals.com":  "Photoreal retail & ecommerce visuals; packages, carts, unbranded products; bright clean light; no logos; no text.",
}
DEFAULT_SITE = "vailvacay.com"

# Read keys from Streamlit Secrets (if provided)
SECRETS = st.secrets.get("api_keys", {})

@dataclass
class Candidate:
    title: str
    source: str
    preview_bytes: bytes
    license_note: str

def slugify(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[’'`]", "", t)
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t

def crop_resize_to(img: Image.Image, w: int, h: int) -> Image.Image:
    t_ratio = w / h
    iw, ih = img.size
    if iw / ih > t_ratio:
        new_w = int(ih * t_ratio)
        x0 = (iw - new_w) // 2
        box = (x0, 0, x0 + new_w, ih)
    else:
        new_h = int(iw / t_ratio)
        y0 = (ih - new_h) // 2
        box = (0, y0, iw, y0 + new_h)
    return img.crop(box).resize((w, h), Image.LANCZOS)

def to_webp_bytes(img_bytes: bytes, w: int, h: int, quality: int) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = crop_resize_to(img, w, h)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=max(60, min(95, quality)), method=6)
    return buf.getvalue()

# -----------------------------
# Google / SerpAPI fetchers
# -----------------------------

def google_textsearch_place(query: str, gmaps_key: str) -> Optional[dict]:
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    r = requests.get(url, params={"query": query, "key": gmaps_key}, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    return (data.get("results") or [None])[0]

def google_place_details(place_id: str, gmaps_key: str) -> dict:
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    r = requests.get(url, params={
        "place_id": place_id,
        "fields": "name,geometry,photos",
        "key": gmaps_key
    }, timeout=30)
    return (r.json() or {}).get("result", {}) if r.status_code == 200 else {}

def google_photo_bytes(photo_ref: str, gmaps_key: str, max_w: int = 1600) -> Optional[bytes]:
    url = "https://maps.googleapis.com/maps/api/place/photo"
    r = requests.get(url, params={"photoreference": photo_ref, "maxwidth": max_w, "key": gmaps_key},
                     timeout=30, allow_redirects=False)
    loc = r.headers.get("Location")
    if loc:
        img = requests.get(loc, timeout=30)
        if img.status_code == 200:
            return img.content
    if r.status_code == 200 and r.content:
        return r.content
    return None

def streetview_bytes(lat: float, lng: float, gmaps_key: str, radius_m: int = 250,
                     size_w: int = 1024, size_h: int = 1024) -> Optional[bytes]:
    meta = requests.get(
        "https://maps.googleapis.com/maps/api/streetview/metadata",
        params={"location": f"{lat},{lng}", "radius": radius_m, "key": gmaps_key},
        timeout=20,
    ).json()
    if meta.get("status") not in ("OK", "ZERO_RESULTS"):
        return None
    if meta.get("status") == "ZERO_RESULTS":
        return None
    r = requests.get(
        "https://maps.googleapis.com/maps/api/streetview",
        params={"location": f"{lat},{lng}", "radius": radius_m,
                "size": f"{size_w}x{size_h}", "key": gmaps_key},
        timeout=30,
    )
    if r.status_code == 200 and r.content:
        return r.content
    return None

def serpapi_images(query: str, serp_key: str, num: int = 4) -> List[Tuple[str, str]]:
    try:
        r = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google", "q": query, "tbm": "isch", "api_key": serp_key},
            timeout=30
        )
        if r.status_code != 200:
            return []
        data = r.json()
        imgs = []
        for itm in (data.get("images_results") or [])[:num]:
            link = itm.get("original") or itm.get("thumbnail")
            src = itm.get("source") or "Google Images via SerpAPI"
            if link:
                imgs.append((src, link))
        return imgs
    except Exception:
        return []

# -----------------------------
# OpenAI image generation
# -----------------------------

def openai_generate_image_b64(prompt: str, size: str, api_key: str) -> bytes:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "gpt-image-1", "prompt": prompt, "size": size, "response_format": "b64_json"}
    url = "https://api.openai.com/v1/images/generations"
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code == 400 and "response_format" in (r.text or ""):
        payload.pop("response_format", None)
        r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
    jd = r.json()
    if "data" in jd and jd["data"]:
        b64 = jd["data"][0].get("b64_json")
        if b64:
            return base64.b64decode(b64)
        url2 = jd["data"][0].get("url")
        if url2:
            img = requests.get(url2, timeout=60)
            if img.status_code == 200:
                return img.content
    raise RuntimeError("OpenAI returned no image data.")

def build_ai_prompt(site: str, keyword: str) -> str:
    base = SITE_PROFILES.get(site, SITE_PROFILES[DEFAULT_SITE])
    k = keyword.lower()
    style_hints = []
    if any(x in k for x in ["ski", "snow", "back bowl", "vail in january", "winter"]):
        style_hints.append("winter setting; snow present where appropriate")
    if any(x in k for x in ["summer", "july", "august"]):
        style_hints.append("summer setting; green trees, bright light")
    if any(x in k for x in ["november", "october"]):
        style_hints.append("shoulder season feel; transitional foliage")
    style = ", ".join(style_hints) if style_hints else "scene appropriate to the topic"
    return (f"{base} Create a photorealistic landscape-orientation image for: '{keyword}'. "
            f"Balanced composition; natural light; editorial stock-photo feel; "
            f"no text or logos; no brand marks. Scene intent: {style}.")

# -----------------------------
# Real-photo candidate collector
# -----------------------------

def collect_real_photo_candidates(q: str,
                                  use_places_flag: bool,
                                  use_street_flag: bool,
                                  use_serp_flag: bool,
                                  sv_radius_m: int,
                                  gmaps_key: str,
                                  serp_key: Optional[str]) -> List[Candidate]:
    cands: List[Candidate] = []
    if not gmaps_key:
        return cands

    place = google_textsearch_place(q, gmaps_key)
    if not place:
        return cands

    details = google_place_details(place["place_id"], gmaps_key)
    title = details.get("name") or q
    loc = (details.get("geometry") or {}).get("location") or {}
    lat, lng = loc.get("lat"), loc.get("lng")

    # Google Places Photos
    if use_places_flag:
        for ph in (details.get("photos") or [])[:12]:
            ref = ph.get("photo_reference")
            if not ref:
                continue
            try:
                img_bytes = google_photo_bytes(ref, gmaps_key, max_w=1600)
                if img_bytes:
                    cands.append(Candidate(
                        title=f"Google Places Photo — {title}",
                        source="Google Maps contributor",
                        preview_bytes=img_bytes,
                        license_note="License: Refer to Google Places Photo terms",
                    ))
            except Exception:
                pass

    # Google Street View
    if use_street_flag and lat and lng:
        try:
            sv = streetview_bytes(lat, lng, gmaps_key, radius_m=sv_radius_m, size_w=1024, size_h=1024)
            if sv:
                cands.append(Candidate(
                    title=f"Google Street View — {title}",
                    source="Google Street View",
                    preview_bytes=sv,
                    license_note="License: Refer to Google Street View terms",
                ))
        except Exception:
            pass

    # SerpAPI (reference-only)
    if use_serp_flag and serp_key:
        refs = serpapi_images(q, serp_key, num=6)
        for src, url in refs:
            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 200 and r.content:
                    cands.append(Candidate(
                        title=f"SerpAPI (Google Images, reference) — {title}",
                        source=src + " (reference only)",
                        preview_bytes=r.content,
                        license_note="License: Unknown / reference-only",
                    ))
            except Exception:
                pass

    return cands

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(f"{APP_NAME} — Real Photos + AI Render + Excel Batch")

# Mode
mode = st.sidebar.radio("Mode", ["Real Photos", "AI Render", "Batch from Excel"], index=0)

# Keys (with Secrets fallback)
st.sidebar.subheader("Keys")
gmaps_key_input = st.sidebar.text_input("Google Maps/Places API key", type="password")
serp_key_input  = st.sidebar.text_input("SerpAPI key (optional)", type="password")
openai_key_input = st.sidebar.text_input("OpenAI API key (for AI Render)", type="password")
gmaps_key  = gmaps_key_input  or SECRETS.get("GOOGLE_MAPS_API_KEY", "")
serp_key   = serp_key_input   or SECRETS.get("SERPAPI_KEY", "")
openai_key = openai_key_input or SECRETS.get("OPENAI_API_KEY", "")

# Output
st.sidebar.subheader("Output")
quality = st.sidebar.slider("WebP quality", 60, 95, 82)
make_pin = st.sidebar.checkbox("Also make a Pinterest image (1000×1500)")

# AI settings
st.sidebar.subheader("AI settings")
site = st.sidebar.selectbox("Site style", list(SITE_PROFILES.keys()),
                            index=list(SITE_PROFILES.keys()).index(DEFAULT_SITE))

# LSI controls (AI only)
st.sidebar.caption("LSI expansion is used by AI mode.")
lsi_method = st.sidebar.selectbox("LSI method", ["Heuristic", "Off"], index=0)
images_per_keyword = st.sidebar.number_input("Images per keyword (LSI expansion)", 1, 10, 1)

# Sources (real photos)
st.sidebar.subheader("Sources to use")
use_places_flag = st.sidebar.checkbox("Google Places Photos", value=True, key="use_places_flag")
use_street_flag = st.sidebar.checkbox("Google Street View", value=True, key="use_street_flag")
use_serp_flag = st.sidebar.checkbox("SerpAPI thumbnails (reference only)", value=False, key="use_serp_flag")

# Street View radius
st.sidebar.subheader("Street View")
sv_radius_m = st.sidebar.slider("Search radius (meters)", 25, 500, 250, key="sv_radius_m")

# Invalidate old candidates if source settings change
sources_sig = (use_places_flag, use_street_flag, use_serp_flag, sv_radius_m)
if st.session_state.get("last_sources_sig") != sources_sig:
    st.session_state["last_sources_sig"] = sources_sig
    st.session_state.pop("realphoto_sets", None)

# Session containers
if "realphoto_sets" not in st.session_state:
    st.session_state["realphoto_sets"] = {}
if "zip_items" not in st.session_state:
    st.session_state["zip_items"] = []  # list of (filename, bytes)

# -----------------------------
# Keyword input or Excel upload
# -----------------------------

if mode in ("Real Photos", "AI Render"):
    keywords_text = st.text_area("Paste keywords (one per line)", height=140,
                                 placeholder="Tavern on the Square, Vail Colorado\nBest seafood restaurant in Boston")
    col_a, col_b = st.columns([1, 1])

if mode == "Real Photos":
    if col_a.button("Collect candidates"):
        st.session_state["realphoto_sets"] = {}
        kws = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
        if not kws:
            st.warning("Please paste at least one keyword.")
        else:
            if not gmaps_key:
                st.error("Please enter your Google Maps/Places API key in the sidebar.")
            else:
                prog = st.progress(0.0)
                for i, kw in enumerate(kws, start=1):
                    st.session_state["realphoto_sets"][kw] = collect_real_photo_candidates(
                        kw, use_places_flag, use_street_flag, use_serp_flag, sv_radius_m,
                        gmaps_key, serp_key if use_serp_flag else None
                    )
                    prog.progress(i / len(kws))
                st.success("Candidates collected. Scroll down and click **Create Image** on any thumbnail.")
    if col_b.button("Clear"):
        st.session_state["realphoto_sets"].clear()
        st.session_state["zip_items"].clear()
        st.success("Cleared.")

    sets = st.session_state.get("realphoto_sets", {})
    for kw, cands in sets.items():
        if not use_serp_flag:
            cands = [c for c in cands if "SerpAPI" not in c.title]

        st.markdown(f"### {kw}")
        if not cands:
            st.info("No candidates found yet. Try a different query or increase Street View radius.")
            continue

        cols = st.columns(3)
        for idx, c in enumerate(cands):
            with cols[idx % 3]:
                st.image(c.preview_bytes, use_container_width=True, caption=c.title)
                st.caption(c.license_note)
                st.caption(f"Credit: {c.source}")

                fn = f"{slugify(kw)}_{idx}.webp"
                if st.button("Create Image", key=f"create_{kw}_{idx}"):
                    try:
                        webp = to_webp_bytes(c.preview_bytes, OUTPUT_W, OUTPUT_H, quality)
                        st.session_state["zip_items"].append((fn, webp))
                        st.success(f"Created {fn}")
                        st.download_button("Download", data=webp, file_name=fn, mime="image/webp",
                                           key=f"dl_{kw}_{idx}")
                        if make_pin:
                            pin_fn = f"{slugify(kw)}_{idx}_pinterest.webp"
                            pin_b = to_webp_bytes(c.preview_bytes, PINTEREST_W, PINTEREST_H, quality)
                            st.session_state["zip_items"].append((pin_fn, pin_b))
                            st.download_button("Download Pinterest", data=pin_b, file_name=pin_fn,
                                               mime="image/webp", key=f"dlp_{kw}_{idx}")
                    except Exception as e:
                        st.error(f"Failed: {e}")

    if st.session_state["zip_items"]:
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fn, b in st.session_state["zip_items"]:
                zf.writestr(fn, b)
        buf.seek(0)
        st.download_button("⬇️ Download all as ZIP", data=buf, file_name="imageforge_realphotos.zip",
                           mime="application/zip")

elif mode == "AI Render":
    size = st.selectbox("OpenAI render size", OPENAI_IMAGE_SIZES, index=0)
    if col_a.button("Generate Image"):
        if not openai_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        else:
            kws = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
            if not kws:
                st.warning("Please paste at least one keyword.")
            else:
                outputs: List[Tuple[str, bytes]] = []
                prog = st.progress(0.0)
                total = 0
                for kw in kws:
                    variants = [kw]
                    if images_per_keyword > 1 and lsi_method == "Heuristic":
                        base = kw
                        hints = [
                            "at golden hour", "wide angle", "close-up details",
                            "with people candidly present", "without people",
                            "from a high vantage", "street-level perspective",
                            "moody overcast light", "bright clear sky"
                        ]
                        for h in hints[:images_per_keyword - 1]:
                            variants.append(f"{base} — {h}")
                    for v in variants:
                        try:
                            prompt = build_ai_prompt(site, v)
                            png = openai_generate_image_b64(prompt, size, openai_key)
                            webp = to_webp_bytes(png, OUTPUT_W, OUTPUT_H, quality)
                            fn = f"{slugify(v)}.webp"
                            outputs.append((fn, webp))
                            st.image(webp, caption=fn, use_container_width=True)
                            st.download_button("Download", data=webp, file_name=fn,
                                               mime="image/webp", key=f"dl_ai_{fn}")
                            if make_pin:
                                pin_fn = f"{slugify(v)}_pinterest.webp"
                                pin_b = to_webp_bytes(png, PINTEREST_W, PINTEREST_H, quality)
                                st.download_button("Download Pinterest", data=pin_b, file_name=pin_fn,
                                                   mime="image/webp", key=f"dlp_ai_{pin_fn}")
                        except Exception as e:
                            st.error(f"{v}: {e}")
                        total += 1
                        prog.progress(min(1.0, total / max(1, (len(kws) * images_per_keyword))))
                if outputs:
                    import zipfile
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for fn, b in outputs:
                            zf.writestr(fn, b)
                    buf.seek(0)
                    st.download_button("⬇️ Download all as ZIP", data=buf, file_name="imageforge_ai.zip",
                                       mime="application/zip")
    if col_b.button("Clear"):
        st.experimental_rerun()

# -----------------------------
# New: Batch from Excel
# -----------------------------
else:
    st.markdown("### Upload an Excel file with business names")
    st.write("Accepted: `.xlsx` (first row as headers). Common header names: **name**, **business**, **place**, or choose any column below.")

    file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    if file:
        try:
            df = pd.read_excel(file)
        except Exception as e:
            st.error(f"Unable to read Excel: {e}")
            st.stop()

        if df.empty:
            st.warning("The spreadsheet is empty.")
            st.stop()

        # Pick a column for queries
        default_col = None
        for c in df.columns:
            if str(c).strip().lower() in ("name", "business", "place", "query", "keyword"):
                default_col = c
                break
        column = st.selectbox("Select the column that contains business names/queries:", list(df.columns),
                              index=(list(df.columns).index(default_col) if default_col in df.columns else 0))

        # Optional: city/state context to improve disambiguation
        context_hint = st.text_input("Optional location/context to append to each query (e.g., 'Vail, Colorado')", "")

        # Batch options
        col1, col2 = st.columns(2)
        max_places_photos = col1.number_input("Max Places Photos to try per business", 1, 12, 6)
        use_street_in_batch = col2.checkbox("Also try Street View if no Places Photo", value=True)

        start = st.button("Run batch (Create images + ZIP)")
        if start:
            if not gmaps_key:
                st.error("Google Maps/Places API key is required (enter in sidebar).")
                st.stop()

            import zipfile
            zip_buf = io.BytesIO()
            created = 0
            prog = st.progress(0.0)

            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                rows = df[column].astype(str).tolist()
                total = len(rows)
                for i, raw in enumerate(rows, start=1):
                    q = raw.strip()
                    if not q:
                        prog.progress(i/total); continue
                    if context_hint:
                        q = f"{q}, {context_hint}"

                    # Collect candidates (limit Places photos in this batch)
                    cands = collect_real_photo_candidates(
                        q,
                        use_places_flag=True,
                        use_street_flag=use_street_in_batch,
                        use_serp_flag=False,             # keep Serp off in batch
                        sv_radius_m=sv_radius_m,
                        gmaps_key=gmaps_key,
                        serp_key=None
                    )

                    # Prefer Places Photo; fall back to Street View
                    chosen = None
                    places = [c for c in cands if "Places Photo" in c.title]
                    street = [c for c in cands if "Street View" in c.title]
                    pool = (places[:max_places_photos] or street[:1])

                    if pool:
                        chosen = pool[0]

                    if chosen:
                        try:
                            webp = to_webp_bytes(chosen.preview_bytes, OUTPUT_W, OUTPUT_H, quality)
                            base = slugify(raw)
                            fn = f"{base or 'image'}_{i}.webp"
                            zf.writestr(fn, webp)
                            created += 1
                            if make_pin:
                                pin_b = to_webp_bytes(chosen.preview_bytes, PINTEREST_W, PINTEREST_H, quality)
                                zf.writestr(f"{base or 'image'}_{i}_pinterest.webp", pin_b)
                        except Exception as e:
                            st.write(f"⚠️ {raw}: {e}")

                    prog.progress(i/total)

            zip_buf.seek(0)
            st.success(f"Done. Created {created} images.")
            st.download_button("⬇️ Download batch as ZIP", data=zip_buf,
                               file_name="imageforge_batch.zip", mime="application/zip")
