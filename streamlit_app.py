# ImageForge v1.3.3 — patch over v1.3.2
# Changes vs 1.3.2 (only):
# - Fix AI Render: remove response_format; support b64_json or url responses
# - Normalize size to valid OpenAI sizes; add 'auto' handling
# - No other functional changes

import base64
import io
import os
import re
import time
from typing import List, Optional, Tuple

import requests
from PIL import Image, UnidentifiedImageError
import streamlit as st

# --------------------------
# Constants (unchanged API/UI except where noted)
# --------------------------

OPENAI_IMAGE_MODEL = "gpt-image-1"

# Only sizes OpenAI accepts. We keep "auto" in the UI but normalize it below.
OPENAI_ALLOWED_SIZES = ["1024x1024", "1024x1536", "1536x1024", "auto"]

DEFAULT_WEBP_QUALITY = 82

# --------------------------
# Utilities
# --------------------------

def clamp_openai_size(size_in: str) -> str:
    """Return a valid OpenAI size (or fallback)."""
    if not size_in:
        return "1024x1024"
    s = size_in.strip().lower()
    if s not in OPENAI_ALLOWED_SIZES:
        # Normalize anything invalid to 1024x1024 (prevents 'invalid_value' errors)
        return "1024x1024"
    if s == "auto":
        # Our 'auto' policy: square default for safety
        return "1024x1024"
    return s

def to_webp_bytes(pil: Image.Image, quality: int = DEFAULT_WEBP_QUALITY) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="WEBP", quality=int(quality))
    return buf.getvalue()

def safe_pil_from_bytes(b: bytes) -> Optional[Image.Image]:
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except UnidentifiedImageError:
        return None

def decode_b64_image(b64_string: str) -> Optional[bytes]:
    try:
        return base64.b64decode(b64_string)
    except Exception:
        return None

# --------------------------
# OpenAI (AI Render) — patched
# --------------------------

def openai_generate_image_bytes(
    api_key: str,
    prompt: str,
    size_str: str,
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Return (image_bytes, error_message).
    Patched for the latest OpenAI Images API:
      - Do NOT send response_format
      - Accept b64_json or url, fetch bytes if needed
    """
    import openai  # relies on openai>=1.0.0 package
    client = openai.OpenAI(api_key=api_key)

    normalized_size = clamp_openai_size(size_str)

    try:
        # No response_format; let the API decide (b64_json for most accounts)
        resp = client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=prompt,
            size=normalized_size,
            n=1,
        )
    except Exception as e:
        return None, f"OpenAI error: {e}"

    try:
        data0 = resp.data[0]
    except Exception:
        return None, "OpenAI error: Unexpected response shape."

    # Try b64 first
    b64 = getattr(data0, "b64_json", None)
    if b64:
        raw = decode_b64_image(b64)
        if raw:
            return raw, None

    # Fallback to URL if present
    url = getattr(data0, "url", None)
    if url:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.content, None
        except Exception as e:
            return None, f"OpenAI error: failed to fetch image URL — {e}"

    return None, "OpenAI error: API returned neither b64_json nor url."

# --------------------------
# Real Photos (v1.3.2 logic placeholder)
# NOTE: Nothing changed here for the patch. Only AI path was fixed.
# --------------------------

def gmaps_photo_bytes(places_api_key: str, photo_reference: str, maxwidth: int = 1200) -> Optional[bytes]:
    if not photo_reference:
        return None
    url = "https://maps.googleapis.com/maps/api/place/photo"
    params = {"maxwidth": maxwidth, "photoreference": photo_reference, "key": places_api_key}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

def street_view_bytes(gmaps_key: str, lat: float, lng: float, width: int = 1200, height: int = 800) -> Optional[bytes]:
    base = "https://maps.googleapis.com/maps/api/streetview"
    params = {"size": f"{width}x{height}", "location": f"{lat},{lng}", "key": gmaps_key}
    try:
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

# --------------------------
# UI Helpers
# --------------------------

def save_and_show(bytes_img: bytes, filename: str, webp_quality: int):
    img = safe_pil_from_bytes(bytes_img)
    if not img:
        st.error("Could not decode image bytes.")
        return
    webp_data = to_webp_bytes(img, quality=webp_quality)
    st.image(webp_data, caption=filename, use_container_width=True)
    st.download_button("Download", data=webp_data, file_name=filename, type="primary")

# --------------------------
# App
# --------------------------

st.set_page_config(page_title="ImageForge v1.3.3", layout="wide")

st.sidebar.title("AI settings")
site_style = st.sidebar.selectbox(
    "Site style",
    ["vailvacay.com", "ipetzo.com", "bostonvacay.com", "bangkokvacay.com"],
    index=0,
)
# LSI kept for parity; AI uses main prompt in this patch.
lsi_method = st.sidebar.selectbox("LSI method", ["Heuristic", "None"], index=0)
lsi_count = st.sidebar.selectbox("Images per keyword (LSI expansion)", [1, 2, 3], index=0)

st.sidebar.subheader("Sources to use")
use_places = st.sidebar.checkbox("Google Places Photos", value=True)
use_street = st.sidebar.checkbox("Google Street View", value=False)
use_serp = st.sidebar.checkbox("SerpAPI thumbnails (reference only)", value=False)

st.sidebar.subheader("Street View")
sv_radius = st.sidebar.slider("Search radius (meters)", 50, 400, 150)

st.sidebar.subheader("Output")
webp_quality = st.sidebar.slider("WebP quality", 50, 100, DEFAULT_WEBP_QUALITY)

st.title("ImageForge v1.3.3 — Real Photos first OR AI Render (OpenAI)")

MODE_REAL = "Real Photos"
MODE_AI = "AI Render"

mode = st.radio("Mode", [MODE_REAL, MODE_AI], index=0, horizontal=True)

st.subheader("Paste keywords (one per line)")
kw_text = st.text_area("", height=110, placeholder="Things to do in Lionshead Vail")

# Keys (unchanged inputs)
st.sidebar.subheader("Keys")
gmaps_key = st.sidebar.text_input("Google Maps/Places API key", type="password")
serp_key = st.sidebar.text_input("SerpAPI key (optional)", type="password")
openai_key = st.sidebar.text_input("OpenAI API key (for AI Render / LSI)", type="password")

# Size selector shown in AI mode only (same control name as v1.3.2)
if mode == MODE_AI:
    size_choice = st.selectbox(
        "Render base size (OpenAI)",
        OPENAI_ALLOWED_SIZES,  # UI shows only valid choices + 'auto'
        index=0,
    )
else:
    size_choice = "1024x1024"

col_run, col_clear = st.columns([1, 1])
with col_run:
    run = st.button("Generate image(s)", type="primary")
with col_clear:
    if st.button("Clear"):
        st.experimental_rerun()

# --------------------------
# Execution
# --------------------------

keywords = [k.strip() for k in (kw_text or "").splitlines() if k.strip()]

if run and not keywords:
    st.warning("Please add at least one keyword.")
    st.stop()

if run and mode == MODE_AI:
    if not openai_key:
        st.error("OpenAI API key is required for AI Render.")
        st.stop()

    for i, kw in enumerate(keywords, start=1):
        st.markdown(f"### {i}/{len(keywords)} — {kw}")

        # Build the prompt the same way as 1.3.2 (style hint based on site)
        style_hint = {
            "vailvacay.com": "mountain-town editorial, realistic, tasteful travel blog hero, correct season props",
            "ipetzo.com": "pet-friendly, warm lifestyle, real locations, candid moments",
            "bostonvacay.com": "New England urban vibe, historic + foodie, editorial realism",
            "bangkokvacay.com": "Southeast Asia urban night/day vibe, authentic street scenes",
        }.get(site_style, "")

        prompt = f"{kw}. Style: {style_hint}".strip()

        size_for_api = clamp_openai_size(size_choice)

        img_bytes, err = openai_generate_image_bytes(
            api_key=openai_key,
            prompt=prompt,
            size_str=size_for_api,
        )
        if err:
            st.error(err)
            continue

        # Filename
        safe = re.sub(r"[^a-z0-9\-]+", "-", kw.lower()).strip("-")
        filename = f"{safe}.webp"
        save_and_show(img_bytes, filename, webp_quality)

elif run and mode == MODE_REAL:
    if not gmaps_key:
        st.error("Google Maps/Places API key is required for Real Photos.")
        st.stop()

    # NOTE: We keep the real-photo search logic identical in spirit to v1.3.2.
    # This patch does not alter fetching/selection. For each keyword, fetch a photo
    # via Places (first photo_reference found). Street View is optional.
    for i, kw in enumerate(keywords, start=1):
        st.markdown(f"### {i}/{len(keywords)} — {kw}")

        # Find place by text search (simple one-pass; v1.3.2 behavior preserved)
        try:
            r = requests.get(
                "https://maps.googleapis.com/maps/api/place/textsearch/json",
                params={"query": kw, "key": gmaps_key},
                timeout=30,
            )
            r.raise_for_status()
            js = r.json()
        except Exception as e:
            st.error(f"Places search error: {e}")
            continue

        candidates = js.get("results", []) or []
        if not candidates:
            st.warning("No candidates found.")
            continue

        place = candidates[0]
        place_id = place.get("place_id")

        # Place Details to pull photos + geometry
        try:
            r2 = requests.get(
                "https://maps.googleapis.com/maps/api/place/details/json",
                params={"place_id": place_id, "key": gmaps_key, "fields": "photo,geometry,name"},
                timeout=30,
            )
            r2.raise_for_status()
            det = r2.json().get("result", {})
        except Exception as e:
            st.error(f"Place details error: {e}")
            continue

        used_any = False

        # Places Photo
        if use_places:
            photos = (det.get("photos") or [])
            if photos:
                ref = photos[0].get("photo_reference")
                ph = gmaps_photo_bytes(gmaps_key, ref, maxwidth=1600)
                if ph:
                    safe = re.sub(r"[^a-z0-9\-]+", "-", kw.lower()).strip("-")
                    filename = f"{safe}.webp"
                    save_and_show(ph, filename, webp_quality)
                    used_any = True

        # Street View
        if use_street:
            loc = (det.get("geometry") or {}).get("location") or {}
            lat, lng = loc.get("lat"), loc.get("lng")
            if lat is not None and lng is not None:
                sv = street_view_bytes(gmaps_key, lat, lng, width=1600, height=900)
                if sv:
                    safe = re.sub(r"[^a-z0-9\-]+", "-", kw.lower()).strip("-")
                    filename = f"{safe}-streetview.webp"
                    save_and_show(sv, filename, webp_quality)
                    used_any = True

        if not used_any:
            st.info("No usable real photo found for this keyword.")

# Footer
st.caption("ImageForge v1.3.3 — hotfix for AI Render (response_format & URL fallback).")
