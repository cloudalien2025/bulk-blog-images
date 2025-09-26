# ImageForge v1.3.1a (hotfix: Real Photos button text)
# - Keeps v1.3.1 behavior and options
# - Real Photos: primary button reads "Collect candidates"
# - AI Render: primary button reads "Generate image(s)"
# - Web Story (1080x1920) export option retained
# - Robust OpenAI image generation (accepts url or b64_json; no response_format)

import os
import io
import math
import base64
import zipfile
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import requests
from PIL import Image, ImageOps
import streamlit as st

# ---------------------------
# Constants / Defaults
# ---------------------------

APP_TITLE = "ImageForge v1.3.1a"
OPENAI_MODEL = "gpt-image-1"

OPENAI_ALLOWED_SIZES = ["1024x1024", "1024x1536", "1536x1024", "auto"]
# We'll optionally publish "Web Story" 1080x1920 on export by resampling.

SITE_STYLES = [
    "vailvacay.com",
    "bangkokvacay.com",
    "bostonvacay.com",
    "ipetzo.com",
]

LSI_METHODS = ["Heuristic", "None"]

# ---------------------------
# Utilities
# ---------------------------

def _bytes_to_image(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def _fetch_bytes(url: str, timeout: int = 20) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception:
        return None

def _save_webp(img: Image.Image, quality: int = 82) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=6)
    return buf.getvalue()

def _resize_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Like CSS background-size: cover; center crop."""
    src_w, src_h = img.size
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # too wide -> height-fit then crop sides
        new_h = target_h
        new_w = int(round(new_h * src_ratio))
    else:
        # too tall -> width-fit then crop top/bottom
        new_w = target_w
        new_h = int(round(new_w / src_ratio))

    img2 = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    return img2.crop((left, top, right, bottom))

def _today_season_hint(lat: float, lon: float) -> str:
    """Very light seasonal nudge based on hemisphere."""
    # Northern hemisphere if lat >= 0
    month = datetime.utcnow().month
    if lat >= 0:
        # simple mapping
        if month in (12,1,2):
            return "It is winter in this location."
        if month in (3,4,5):
            return "It is spring in this location."
        if month in (6,7,8):
            return "It is summer in this location."
        return "It is autumn in this location."
    else:
        # southern hemisphere flipped
        if month in (12,1,2):
            return "It is summer in this location."
        if month in (3,4,5):
            return "It is autumn in this location."
        if month in (6,7,8):
            return "It is winter in this location."
        return "It is spring in this location."

# ---------------------------
# Google Places / Street View / SerpAPI helpers
# ---------------------------

def google_find_place(api_key: str, query: str) -> Optional[dict]:
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": query,
        "inputtype": "textquery",
        "fields": "place_id,name,formatted_address,geometry",
        "key": api_key
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        j = r.json()
        cands = j.get("candidates", [])
        if cands:
            return cands[0]
        return None
    except Exception:
        return None

def google_place_photos(api_key: str, place_id: str) -> List[Dict]:
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,photo,photos",
        "key": api_key
    }
    out = []
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        j = r.json()
        result = j.get("result", {})
        photos = result.get("photos", []) or []
        for p in photos[:8]:
            ref = p.get("photo_reference")
            if not ref:
                continue
            # Google Places Photos endpoint URL (we'll fetch bytes client-side)
            photo_url = (
                "https://maps.googleapis.com/maps/api/place/photo"
                f"?maxwidth=1600&photo_reference={ref}&key={api_key}"
            )
            out.append({
                "title": result.get("name", "Google Places Photo"),
                "source": "Google Places Photo",
                "preview_url": photo_url,
                "credit": "Google Maps contributor",
                "license": "Refer to Google Places Photo terms"
            })
    except Exception:
        pass
    return out

def google_street_view_candidate(api_key: str, lat: float, lon: float, radius_m: int) -> Optional[Dict]:
    """
    We try metadata first to see if a pano exists within radius. If yes, construct a static API URL.
    """
    meta_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "location": f"{lat},{lon}",
        "radius": radius_m,
        "key": api_key
    }
    try:
        rm = requests.get(meta_url, params=params, timeout=20)
        rm.raise_for_status()
        jm = rm.json()
        if jm.get("status") == "OK":
            # static image request
            # note size is limited by API (640 default; scale=2 returns 1280 width)
            static_url = (
                "https://maps.googleapis.com/maps/api/streetview"
                f"?size=640x640&scale=2&location={lat},{lon}&fov=80&key={api_key}"
            )
            return {
                "title": "Google Street View",
                "source": "Google Street View",
                "preview_url": static_url,
                "credit": "Google Street View",
                "license": "Refer to Google Street View terms"
            }
        return None
    except Exception:
        return None

def serpapi_image_refs(serp_key: str, query: str, num: int = 4) -> List[Dict]:
    out = []
    try:
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google",
            "q": query,
            "tbm": "isch",
            "ijn": "0",
            "api_key": serp_key
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        j = r.json()
        imgs = j.get("images_results", [])[:num]
        for im in imgs:
            link = im.get("original") or im.get("thumbnail")
            if not link:
                continue
            out.append({
                "title": im.get("title") or "Google Images (ref)",
                "source": "SerpAPI (Google Images, reference)",
                "preview_url": link,
                "credit": "Google Images via SerpAPI — reference only",
                "license": "Unknown / reference-only"
            })
    except Exception:
        pass
    return out

# ---------------------------
# OpenAI image generation
# ---------------------------

def openai_generate_images(api_key: str, prompt: str, size: str) -> Optional[bytes]:
    """
    Robust handler:
    - no response_format (per new API)
    - handle either 'url' or 'b64_json' in data[0]
    Return raw image bytes (PNG) or None.
    """
    try:
        url = "https://api.openai.com/v1/images/generations"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": OPENAI_MODEL,
            "prompt": prompt,
        }
        if size != "auto":
            payload["size"] = size

        r = requests.post(url, json=payload, headers=headers, timeout=60)
        # Raise for 4xx/5xx
        if r.status_code >= 400:
            # Streamlit-visible error
            st.error(f"OpenAI error: {r.text}")
            return None

        j = r.json()
        data = j.get("data", [])
        if not data:
            return None

        item = data[0]
        if "b64_json" in item and item["b64_json"]:
            raw = base64.b64decode(item["b64_json"])
            return raw
        if "url" in item and item["url"]:
            b = _fetch_bytes(item["url"])
            return b
        # nothing usable
        st.error("OpenAI error: OpenAI returned neither b64_json nor url.")
        return None
    except Exception as e:
        st.error(f"OpenAI request failed: {e}")
        return None

# ---------------------------
# Session state
# ---------------------------

if "candidates" not in st.session_state:
    st.session_state["candidates"] = {}  # {keyword: [dicts]}
if "generated" not in st.session_state:
    st.session_state["generated"] = []   # list of tuples (filename, bytes)

# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title=APP_TITLE, page_icon="🖼️", layout="wide")
st.title(APP_TITLE)
st.caption("Real photos first (Places/Street View + optional SerpAPI refs) or AI Render via OpenAI. Exports WebP (+ optional Pinterest/Web Story).")

# Mode
mode = st.sidebar.radio("Mode", ["Real Photos", "AI Render"], index=0)

# Keys
st.sidebar.subheader("Keys")
gmaps_key = st.sidebar.text_input("Google Maps/Places API key", type="password", help="Enable Places API + Street View Static API in your Google Cloud project.")
serp_key = st.sidebar.text_input("SerpAPI key (optional)", type="password")
openai_key = st.sidebar.text_input("OpenAI API key (for AI Render)", type="password")

# Output / site
st.sidebar.subheader("Output")
webp_quality = st.sidebar.slider("WebP quality", 60, 100, 82)

# AI-only settings block
st.sidebar.subheader("AI settings")
site_style = st.sidebar.selectbox("Site style", SITE_STYLES)
lsi_method = st.sidebar.selectbox("LSI method", LSI_METHODS)
images_per_keyword = st.sidebar.selectbox("Images per keyword (LSI expansion)", [1, 2, 3], index=0)

# Sources & Street View (Real Photos)
st.sidebar.subheader("Sources to use")
use_places = st.sidebar.checkbox("Google Places Photos", True)
use_street = st.sidebar.checkbox("Google Street View", True)
use_serp = st.sidebar.checkbox("SerpAPI thumbnails (reference only)", False)

st.sidebar.subheader("Street View")
sv_radius = st.sidebar.slider("Search radius (meters)", 50, 400, 150)

# Export helpers
st.sidebar.subheader("Extras")
make_pinterest = st.sidebar.checkbox("Also make a Pinterest image (1000×1500)")
make_web_story = st.sidebar.checkbox("Also make a Web Story image (1080×1920)")

# Main input
keywords = st.text_area(
    "Paste keywords (one per line)",
    placeholder="e.g. Things to do in Lionshead Vail\nTavern on the Square, Vail Colorado",
    height=120
)

# Render base size (OpenAI)
size_choice = st.selectbox(
    "Render base size (OpenAI)",
    OPENAI_ALLOWED_SIZES,
    index=2,
    help="OpenAI supports 1024×1024, 1024×1536, 1536×1024, or 'auto'. Web Story export upscales to 1080×1920."
)

# Primary button text depends on mode (HOTFIX)
if mode == "Real Photos":
    main_btn_text = "Collect candidates"
else:
    main_btn_text = "Generate image(s)"

col_btn, col_clear = st.columns([1, 1])
go = col_btn.button(main_btn_text, type="primary")
clear = col_clear.button("Clear")

if clear:
    st.session_state["candidates"].clear()
    st.session_state["generated"].clear()
    st.rerun()

# ---------------------------
# PROCESS
# ---------------------------

kw_list = [k.strip() for k in keywords.splitlines() if k.strip()]

def _export_and_show(img: Image.Image, base_filename: str):
    # Master WebP
    base_webp = _save_webp(img, quality=webp_quality)
    st.image(img, caption=base_filename + ".webp", use_container_width=True)
    st.download_button(
        "Download",
        data=base_webp,
        file_name=base_filename + ".webp",
        mime="image/webp",
    )
    st.session_state["generated"].append((base_filename + ".webp", base_webp))

    # Pinterest (1000×1500)
    if make_pinterest:
        pimg = _resize_cover(img, 1000, 1500)
        pbytes = _save_webp(pimg, quality=webp_quality)
        st.download_button(
            "Download Pinterest",
            data=pbytes,
            file_name=base_filename + "_pinterest_1000x1500.webp",
            mime="image/webp",
        )
        st.session_state["generated"].append((base_filename + "_pinterest_1000x1500.webp", pbytes))

    # Web Story (1080×1920)
    if make_web_story:
        ws = _resize_cover(img, 1080, 1920)
        wsbytes = _save_webp(ws, quality=webp_quality)
        st.download_button(
            "Download Web Story",
            data=wsbytes,
            file_name=base_filename + "_webstory_1080x1920.webp",
            mime="image/webp",
        )
        st.session_state["generated"].append((base_filename + "_webstory_1080x1920.webp", wsbytes))

def _site_prompt_prefix(site: str) -> str:
    if site == "vailvacay.com":
        return "Photo-real Vail, Colorado vibe; alpine village, aspens, gondolas, mountains; editorial blog cover."
    if site == "bangkokvacay.com":
        return "Photo-real Bangkok vibe; urban streets, markets, temples, neon; editorial travel cover."
    if site == "bostonvacay.com":
        return "Photo-real Boston vibe; historic brick, brownstones, waterfront; editorial travel cover."
    if site == "ipetzo.com":
        return "Photo-real pet lifestyle vibe; friendly, upbeat, clean composition; editorial cover."
    return ""

# Action
if go and kw_list:
    if mode == "Real Photos":
        # Validate Google key
        if not gmaps_key:
            st.error("Google Maps/Places API key is required for Real Photos mode.")
        else:
            for idx, kw in enumerate(kw_list, start=1):
                st.markdown(f"### {idx}/{len(kw_list)} — {kw}")

                # Find place
                place = google_find_place(gmaps_key, kw)
                if not place:
                    st.warning("No place found for this query.")
                    continue

                lat = place["geometry"]["location"]["lat"]
                lon = place["geometry"]["location"]["lng"]

                # Gather candidates
                cand_list: List[Dict] = []
                if use_places:
                    pid = place.get("place_id")
                    cand_list.extend(google_place_photos(gmaps_key, pid))

                if use_street:
                    sv = google_street_view_candidate(gmaps_key, lat, lon, sv_radius)
                    if sv:
                        cand_list.append(sv)

                if use_serp and serp_key:
                    cand_list.extend(serpapi_image_refs(serp_key, kw, num=6))

                if not cand_list:
                    st.info("No candidates found. Try widening your query or enabling more sources.")
                    continue

                # Store in session (so subsequent Create Image clicks can read)
                st.session_state["candidates"][kw] = cand_list

    else:
        # AI Render
        if not openai_key:
            st.error("OpenAI API key is required for AI Render.")
        else:
            for idx, kw in enumerate(kw_list, start=1):
                st.markdown(f"### {idx}/{len(kw_list)} — {kw}")

                # Optional LSI (kept lightweight)
                expansions = [kw]
                if images_per_keyword > 1 and lsi_method == "Heuristic":
                    # simplistic expansions
                    base = kw
                    expansions.extend([
                        f"{base} — top sights",
                        f"{base} — local vibe",
                    ][:images_per_keyword-1])

                for j, sub_kw in enumerate(expansions, start=1):
                    with st.spinner(f"Rendering {j}/{len(expansions)}"):
                        # Build prompt
                        prefix = _site_prompt_prefix(site_style)
                        prompt = f"{prefix} {sub_kw}. Ultra realistic, editorial photograph."

                        # NOTE: no response_format; robust url/b64 handler
                        raw_png = openai_generate_images(openai_key, prompt, size_choice)
                        if not raw_png:
                            continue
                        try:
                            img = Image.open(io.BytesIO(raw_png)).convert("RGB")
                        except Exception:
                            st.error("OpenAI returned invalid image data.")
                            continue

                        base_name = sub_kw.lower().strip().replace(" ", "-")
                        base_name = "".join(ch for ch in base_name if ch.isalnum() or ch in "-_")
                        _export_and_show(img, base_name)

# ---------------------------
# Candidate deck (Real Photos)
# ---------------------------
if mode == "Real Photos":
    # Show any candidates gathered in this session
    for kw, cands in st.session_state["candidates"].items():
        st.markdown(f"## Thumbnails — choose any to create a final image: {kw}")
        for i, c in enumerate(cands):
            with st.container(border=True):
                st.markdown(f"**[{i}] {c['source']} — {c['title']}**")
                st.caption(f"License: {c.get('license','')}")
                st.caption(f"Credit: {c.get('credit','')}")
                img_bytes = None

                # Load preview
                if c.get("preview_url"):
                    img_bytes = _fetch_bytes(c["preview_url"])
                    if img_bytes:
                        ok = True
                        # Some Street View returns may not be image — guard
                        try:
                            _ = Image.open(io.BytesIO(img_bytes))
                        except Exception:
                            ok = False
                        if ok:
                            st.image(img_bytes, use_container_width=True)
                        else:
                            st.info("Preview unavailable for this candidate.")
                    else:
                        st.info("Preview unavailable for this candidate.")
                else:
                    st.info("No preview URL.")

                # Action: Create Image (save WebP + optional exports)
                btn_key = f"make_{kw}_{i}"
                if st.button("Create Image", key=btn_key, type="primary", use_container_width=False):
                    if not img_bytes:
                        st.error("No image bytes for this candidate.")
                    else:
                        try:
                            img = _bytes_to_image(img_bytes)
                            base_name = kw.lower().strip().replace(" ", "-")
                            base_name = "".join(ch for ch in base_name if ch.isalnum() or ch in "-_")
                            _export_and_show(img, base_name)
                        except Exception as e:
                            st.error(f"Could not finalize image: {e}")

# ---------------------------
# Final ZIP (if anything generated)
# ---------------------------
if st.session_state["generated"]:
    with io.BytesIO() as zip_buf:
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname, b in st.session_state["generated"]:
                zf.writestr(fname, b)
        st.download_button(
            "Download ZIP",
            data=zip_buf.getvalue(),
            file_name="imageforge_export.zip",
            mime="application/zip",
        )
