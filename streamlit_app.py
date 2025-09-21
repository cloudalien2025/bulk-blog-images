# ImageForge v1.3.2 — Real Photos + AI Render
# Patch-only build over v1.3.1:
#   • Centralized, safe OpenAI image generation (no stray response_format; auto-retry).
#   • Allowed OpenAI sizes only. 1536x1024 removed. Clamps any rogue size.
# Everything else (UI/flow/features) remains the same as your baseline.

import os
import io
import re
import json
import base64
import math
import hashlib
import textwrap
import datetime as dt
from typing import List, Dict, Optional, Tuple

import requests
from PIL import Image, ImageOps
import streamlit as st

# -----------------------------
# Constants / Globals
# -----------------------------

APP_TITLE = "ImageForge v1.3.2 — Real Photos + AI Render"
st.set_page_config(page_title=APP_TITLE, layout="wide")

# Only sizes supported by gpt-image-1
ALLOWED_OPENAI_SIZES = {"1024x1024", "1792x1024", "1024x1792"}
RENDER_SIZES = ["1792x1024", "1024x1792", "1024x1024"]  # used in the UI

DEFAULT_WEBP_QUALITY = 82

# -----------------------------
# Small utils
# -----------------------------

def http_get(url: str, timeout: int = 30, headers: Optional[dict] = None) -> bytes:
    r = requests.get(url, timeout=timeout, headers=headers or {})
    r.raise_for_status()
    return r.content

def to_webp_bytes(img_bytes: bytes, target_size: Optional[Tuple[int, int]] = None, quality: int = DEFAULT_WEBP_QUALITY) -> bytes:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if target_size:
        # letterbox/pad to fit Pinterest or similar target while keeping aspect
        im = ImageOps.contain(im, target_size, Image.LANCZOS)
        canvas = Image.new("RGB", target_size, (255, 255, 255))
        off = ((target_size[0] - im.size[0]) // 2, (target_size[1] - im.size[1]) // 2)
        canvas.paste(im, off)
        im = canvas
    out = io.BytesIO()
    im.save(out, format="WEBP", quality=quality, method=6)
    return out.getvalue()

def filename_slug(s: str) -> str:
    slug = re.sub(r"[^a-z0-9\-]+", "-", s.lower().strip().replace("&", "and"))
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "image"

def credit_line(source: str, title: str) -> str:
    return f"{source} — {title}".strip()

# -----------------------------
# OpenAI — single safe entrypoint
# -----------------------------

def openai_images_generate_safe(api_key: str, prompt: str, size: str) -> bytes:
    """
    Single source of truth for OpenAI image generation.
    - clamps size to supported values
    - tries b64_json first
    - if API complains about 'response_format', retries without it and fetches by URL
    """
    size = size if size in ALLOWED_OPENAI_SIZES else "1792x1024"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Attempt 1: request b64_json
    payload = {"model": "gpt-image-1", "prompt": prompt, "size": size, "response_format": "b64_json"}
    r = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=payload, timeout=120)

    if r.status_code == 200:
        data = r.json().get("data") or []
        if data and data[0].get("b64_json"):
            return base64.b64decode(data[0]["b64_json"])

    # Attempt 2: if response_format rejected, retry without it (URL response)
    if r.status_code == 400 and "response_format" in r.text:
        payload2 = {"model": "gpt-image-1", "prompt": prompt, "size": size}
        r2 = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=payload2, timeout=120)
        if r2.status_code != 200:
            raise RuntimeError(f"OpenAI error {r2.status_code}: {r2.text}")
        d2 = r2.json().get("data") or []
        url = d2[0].get("url") if d2 else None
        if not url:
            raise RuntimeError("OpenAI returned neither b64_json nor url.")
        return http_get(url, timeout=120)

    # Otherwise raise the original error so you see what happened
    raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")

# -----------------------------
# Google Places / Street View / SerpAPI (reference)
# -----------------------------

def places_text_search(api_key: str, query: str) -> Optional[dict]:
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": api_key}
    r = requests.get(url, params=params, timeout=40)
    if r.status_code != 200:
        return None
    j = r.json()
    if j.get("results"):
        return j["results"][0]
    return None

def place_details_photos(api_key: str, place_id: str, max_photos: int = 8) -> List[dict]:
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {"place_id": place_id, "key": api_key, "fields": "photo"}
    r = requests.get(url, params=params, timeout=40)
    if r.status_code != 200:
        return []
    j = r.json()
    photos = (j.get("result") or {}).get("photos") or []
    # Normalize
    out = []
    for p in photos[:max_photos]:
        out.append({
            "source": "Google Places Photo",
            "title": "Google Maps contributor",
            "photo_ref": p.get("photo_reference"),
            "width": p.get("width"),
            "height": p.get("height"),
        })
    return out

def places_photo_bytes(api_key: str, photo_ref: str, maxwidth: int = 1600) -> Optional[bytes]:
    url = "https://maps.googleapis.com/maps/api/place/photo"
    params = {"photoreference": photo_ref, "maxwidth": maxwidth, "key": api_key}
    r = requests.get(url, params=params, timeout=60)
    if r.status_code == 200:
        return r.content
    return None

def street_view_find_and_fetch(api_key: str, lat: float, lng: float, radius_m: int = 100) -> Optional[Tuple[bytes, str]]:
    """Find a pano near coords, fetch a static Street View image."""
    meta_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    static_url = "https://maps.googleapis.com/maps/api/streetview"
    params_meta = {"location": f"{lat},{lng}", "radius": radius_m, "key": api_key}
    r = requests.get(meta_url, params=params_meta, timeout=40)
    if r.status_code != 200:
        return None
    j = r.json()
    if j.get("status") != "OK":
        return None
    pano_loc = j.get("location", {})
    # Fetch image (wide-ish)
    params_img = {"size": "1200x800", "location": f"{pano_loc.get('lat')},{pano_loc.get('lng')}", "key": api_key}
    img = requests.get(static_url, params=params_img, timeout=60)
    if img.status_code != 200:
        return None
    return img.content, "Google Street View"

def serpapi_thumbs(serpapi_key: str, query: str, num: int = 4) -> List[dict]:
    url = "https://serpapi.com/search.json"
    params = {"engine": "google", "q": query, "num": num, "tbm": "isch", "api_key": serpapi_key}
    r = requests.get(url, params=params, timeout=40)
    if r.status_code != 200:
        return []
    j = r.json()
    imgs = j.get("images_results") or []
    out = []
    for itm in imgs[:num]:
        out.append({
            "source": "SerpAPI (reference)",
            "title": itm.get("title") or "Google Images via SerpAPI",
            "thumb_url": itm.get("thumbnail") or itm.get("original"),
            "link": itm.get("link"),
        })
    return out

# -----------------------------
# Prompt helpers (kept minimal & unchanged in spirit)
# -----------------------------

SITE_STYLES = {
    "vailvacay.com": "Photoreal Vail/Colorado vibe, alpine village, accurate seasonality and geography.",
    "bostonvacay.com": "Photoreal Boston vibe: brick, brownstones, harbor views, New England seasons.",
    "bangkokvacay.com": "Photoreal Bangkok vibe: tropical light, Thai signage, correct districts.",
    "ipetzo.com": "Pet-centric, friendly, bright, clean lighting, engaging composition.",
}

LSI_METHODS = ["Heuristic", "None"]

def make_prompt(base_kw: str, site_style: str) -> str:
    style = SITE_STYLES.get(site_style, "")
    return textwrap.dedent(f"""
    {base_kw}
    Style: {style}
    Rules:
    - Geographic and season accuracy for the location implied by the keyword.
    - Avoid text/logos unless obviously generic.
    - Crisp, editorial quality; not cartoony.
    """).strip()

# -----------------------------
# UI — Sidebar
# -----------------------------

with st.sidebar:
    st.markdown(f"## Mode")
    mode = st.radio("", ["Real Photos", "AI Render"], index=0)

    st.markdown("## Keys")
    gkey = st.text_input("Google Maps/Places API key", type="password")
    serp_key = st.text_input("SerpAPI key (optional)", type="password")
    openai_key = st.text_input("OpenAI API key (for AI Render / LSI)", type="password")

    st.markdown("## Output")
    webp_quality = st.slider("WebP quality", 60, 95, DEFAULT_WEBP_QUALITY, step=1)

    st.markdown("## AI settings")
    site_style = st.selectbox("Site style", list(SITE_STYLES.keys()), index=0)
    lsi_method = st.selectbox("LSI method", LSI_METHODS, index=0, help="Used by AI mode only.")
    images_per_kw = st.selectbox("Images per keyword (LSI expansion)", [1,2,3,4,5], index=0)

    st.markdown("## Sources to use")
    use_places = st.checkbox("Google Places Photos", value=True)
    use_street = st.checkbox("Google Street View", value=True)
    use_serp = st.checkbox("SerpAPI thumbnails (reference only)", value=False)

    st.markdown("## Street View")
    sv_radius = st.slider("Search radius (meters)", 25, 500, 150, step=25)

    st.markdown("## Pinterest")
    make_pin = st.checkbox("Also make a Pinterest image (1000×1500)")

# -----------------------------
# UI — Main
# -----------------------------

st.title(APP_TITLE)
st.caption("Real photos first (Places/Street View + optional SerpAPI refs) or AI Render via OpenAI. Exports WebP (+ optional Pinterest).")

keywords = st.text_area("Paste keywords (one per line)", height=140, placeholder="e.g.\nTavern on the Square, Vail Colorado\nBest seafood restaurant in Boston\nThings to do in Lionshead Vail")
size = st.selectbox("Render base size (OpenAI)", RENDER_SIZES, index=0)

colA, colB = st.columns([1,1])
btn_generate = colA.button("Generate image(s)")
btn_clear = colB.button("Clear")

if btn_clear:
    st.experimental_rerun()

if not keywords.strip():
    st.stop()

kw_list = [k.strip() for k in keywords.splitlines() if k.strip()]

# -----------------------------
# Real Photos mode
# -----------------------------
if mode == "Real Photos":
    if not gkey:
        st.warning("Enter your Google Maps/Places API key.")
        st.stop()

    for idx, kw in enumerate(kw_list, start=1):
        st.markdown(f"### {idx}/{len(kw_list)} — {kw}")
        place = places_text_search(gkey, kw)
        candidates: List[Dict] = []

        if place:
            pid = place.get("place_id")
            loc = (place.get("geometry") or {}).get("location") or {}
            plat, plng = loc.get("lat"), loc.get("lng")

            if use_places and pid:
                photos = place_details_photos(gkey, pid, max_photos=8)
                # Resolve bytes for preview
                for p in photos:
                    try:
                        pb = places_photo_bytes(gkey, p["photo_ref"], maxwidth=1600)
                        if pb:
                            candidates.append({
                                "source": p["source"],
                                "title": p["title"],
                                "bytes": pb,
                                "refonly": False
                            })
                    except Exception:
                        pass

            if use_street and plat is not None and plng is not None:
                try:
                    sv = street_view_find_and_fetch(gkey, plat, plng, radius_m=sv_radius)
                    if sv:
                        b, src = sv
                        candidates.append({
                            "source": src,
                            "title": kw,
                            "bytes": b,
                            "refonly": False
                        })
                except Exception:
                    pass

        if use_serp and serp_key:
            try:
                for s in serpapi_thumbs(serp_key, kw, num=4):
                    # Reference only: show thumbnail but mark non-downloadable
                    try:
                        tb = http_get(s["thumb_url"], timeout=30)
                        candidates.append({
                            "source": s["source"],
                            "title": s["title"],
                            "bytes": tb,
                            "refonly": True
                        })
                    except Exception:
                        pass
            except Exception:
                pass

        if not candidates:
            st.info("No candidates found.")
            continue

        # Grid of cards with per-card Create Image button
        grid_cols = st.columns(3)
        for i, cand in enumerate(candidates):
            col = grid_cols[i % 3]
            with col:
                st.image(cand["bytes"], use_container_width=True,
                         caption=credit_line(cand["source"], cand["title"]))
                if cand["refonly"]:
                    st.caption("Reference-only (not downloaded).")
                    st.button("Create Image", disabled=True, key=f"refonly_{idx}_{i}")
                else:
                    if st.button("Create Image", key=f"mk_{idx}_{i}"):
                        webp = to_webp_bytes(cand["bytes"], quality=webp_quality)
                        fname = filename_slug(f"{kw}") + ".webp"
                        st.download_button("Download", data=webp, file_name=fname, mime="image/webp")
                        if make_pin:
                            pin = to_webp_bytes(cand["bytes"], target_size=(1000,1500), quality=webp_quality)
                            st.download_button("Download Pinterest", data=pin,
                                               file_name=filename_slug(f"{kw}-pinterest") + ".webp",
                                               mime="image/webp")

# -----------------------------
# AI Render mode
# -----------------------------
else:
    if not openai_key:
        st.warning("Enter your OpenAI API key for AI Render.")
        st.stop()

    for idx, kw in enumerate(kw_list, start=1):
        st.markdown(f"### {idx}/{len(kw_list)} — {kw}")
        n = images_per_kw if images_per_kw else 1
        prompts = []

        # Very light LSI: only if >1 requested and method is heuristic
        if n > 1 and lsi_method == "Heuristic":
            prompts.append(make_prompt(kw, site_style))
            # naive expansions; keep unchanged vs. v1.3.1 spirit
            variations = [
                f"{kw} — wide establishing view",
                f"{kw} — close-up detail",
                f"{kw} — people enjoying the place",
                f"{kw} — golden hour lighting",
                f"{kw} — overcast mood",
                f"{kw} — winter season",
                f"{kw} — summer season",
                f"{kw} — night scene",
            ]
            for v in variations[:(n-1)]:
                prompts.append(make_prompt(v, site_style))
        else:
            prompts = [make_prompt(kw, site_style)]

        out_cols = st.columns(2)
        for j, pr in enumerate(prompts, start=1):
            with out_cols[(j-1) % 2]:
                try:
                    img_bytes = openai_images_generate_safe(openai_key, pr, size)
                except Exception as e:
                    st.error(f"OpenAI error: {e}")
                    continue

                st.image(img_bytes, use_container_width=True, caption=kw)
                webp = to_webp_bytes(img_bytes, quality=webp_quality)
                fname = filename_slug(kw) + ".webp"
                st.download_button("Download", data=webp, file_name=fname, mime="image/webp", key=f"dla_{idx}_{j}")

                if make_pin:
                    pin = to_webp_bytes(img_bytes, target_size=(1000,1500), quality=webp_quality)
                    st.download_button("Download Pinterest", data=pin,
                                       file_name=filename_slug(f"{kw}-pinterest") + ".webp",
                                       mime="image/webp", key=f"dlp_{idx}_{j}")
