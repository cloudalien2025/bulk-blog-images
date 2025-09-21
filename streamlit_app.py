# ImageForge v1.3 — Real Photos + AI Render (Generate Image UX)
# --------------------------------------------------------------
# - Real Photos mode: Google Places Photos + Street View Static (optionally SerpAPI thumbnails as reference-only).
# - AI Render mode: OpenAI Images (gpt-image-1). We ALWAYS call 1024x1024 (API constraint),
#   then crop/resize to your selected output (e.g., 1536x1024) to avoid HTTP 400s.
# - “Generate Image” button appears when AI Render is selected (not “Generate candidates”).
# - Per-thumbnail “Create Image” buttons in Real Photos mode produce WEBP and add to the ZIP queue.
# - Site styles (prompt priors), optional LSI expansion, images-per-keyword, and ZIP download included.
#
# NOTE: This app makes network calls (Google/OpenAI/SerpAPI). Enable those APIs in your accounts.
#       Never hard-code secrets. Paste keys in the sidebar inputs.

import io
import os
import re
import time
import json
import base64
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import requests
from PIL import Image
import streamlit as st

# --------------------------
# App config
# --------------------------
st.set_page_config(page_title="ImageForge v1.3 — Real Photos + AI Render", layout="wide")

# --------------------------
# Constants & site styles
# --------------------------
OUTPUT_DEFAULT_W, OUTPUT_DEFAULT_H = 1200, 675
DEFAULT_WEBP_QUALITY = 82

SITE_PROFILES: Dict[str, str] = {
    "vailvacay.com":  "Photorealistic Colorado Rockies resort & village scenes; alpine rivers; conifers; gondola; tasteful; no brands.",
    "bangkokvacay.com":"Photorealistic Bangkok street/city scenes; temples, neon, night markets, BTS/MRT; tropical light; no brands.",
    "bostonvacay.com": "Photorealistic New England city visuals; brownstones, harbor, brick & cobblestone; foliage; no brands.",
    "ipetzo.com":      "Photorealistic pet lifestyle; dogs/cats with owners at home/park/grooming; warm; no brands/logo text.",
    "1-800deals.com":  "Photorealistic commerce visuals; shopping scenes, parcels; generic packaging; clean backgrounds; no brands.",
}
DEFAULT_SITE = "vailvacay.com"

# --------------------------
# Small helpers
# --------------------------
def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[’'`]", "", s)
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "image"

def crop_to_aspect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    tw, th = target_w, target_h
    target_ratio = tw / th
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

def to_webp_bytes(img_bytes: bytes, w: int, h: int, quality: int) -> bytes:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    im = crop_to_aspect(im, w, h).resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=quality, method=6)
    return buf.getvalue()

def build_prompt(site: str, keyword: str, season_hint: Optional[str]=None) -> str:
    base = SITE_PROFILES.get(site, SITE_PROFILES[DEFAULT_SITE])
    extra = []
    if season_hint:
        extra.append(season_hint)
    # A lightweight topic-to-scene nudge (kept minimal to “let AI be AI”):
    k = keyword.lower()
    if "indoor" in k:
        extra.append("indoor vantage, cozy ambient light")
    if any(x in k for x in ["how far", "between", "to ", "from "]):
        extra.append("travel/wayfinding vibe; sense of route")
    if any(x in k for x in ["hotel", "stay", "resort"]):
        extra.append("inviting lodging exterior or lobby")
    if any(x in k for x in ["restaurant", "best burger", "seafood", "pizza"]):
        extra.append("inviting dining setting; tasteful food context; no brand text")
    # Compose
    extra_txt = (", ".join(extra) + ". ") if extra else ""
    return (
        f"{base} {extra_txt}"
        f"Create an editorial stock-photo style image for: “{keyword}”. "
        f"Balanced composition, natural light, NO words or logos."
    )

# --------------------------
# OpenAI Images (AI Render)
# --------------------------
openai_generate_image_png

# --------------------------
# Google Places / Street View
# --------------------------
GOOGLE_PLACES_TEXTSEARCH = "https://maps.googleapis.com/maps/api/place/textsearch/json"
GOOGLE_PLACES_DETAILS     = "https://maps.googleapis.com/maps/api/place/details/json"
GOOGLE_PHOTO              = "https://maps.googleapis.com/maps/api/place/photo"
GOOGLE_STREETVIEW_STATIC  = "https://maps.googleapis.com/maps/api/streetview"
GOOGLE_STREETVIEW_META    = "https://maps.googleapis.com/maps/api/streetview/metadata"

def google_textsearch_place(query: str, key: str) -> Optional[dict]:
    params = {"query": query, "key": key}
    r = requests.get(GOOGLE_PLACES_TEXTSEARCH, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    return (data.get("results") or [None])[0]

def google_place_details(place_id: str, key: str) -> dict:
    params = {
        "place_id": place_id,
        "key": key,
        "fields": "place_id,name,geometry,photos,url,formatted_address"
    }
    r = requests.get(GOOGLE_PLACES_DETAILS, params=params, timeout=60)
    r.raise_for_status()
    return r.json().get("result", {})

def google_photo_bytes(photo_ref: str, key: str, max_w: int=1600) -> Optional[bytes]:
    # Google redirects to image; requests can follow it.
    params = {"photoreference": photo_ref, "maxwidth": max_w, "key": key}
    r = requests.get(GOOGLE_PHOTO, params=params, timeout=60, allow_redirects=True)
    if r.status_code == 200 and r.content:
        return r.content
    return None

def streetview_pano_exists(lat: float, lng: float, key: str, radius_m: int) -> Optional[dict]:
    params = {"location": f"{lat},{lng}", "radius": radius_m, "key": key}
    r = requests.get(GOOGLE_STREETVIEW_META, params=params, timeout=60)
    r.raise_for_status()
    meta = r.json()
    return meta if meta.get("status") == "OK" else None

def streetview_bytes(lat: float, lng: float, key: str, radius_m: int,
                     size_w: int=1024, size_h: int=1024) -> Optional[bytes]:
    # First verify pano exists (so we can avoid a blank image).
    meta = streetview_pano_exists(lat, lng, key, radius_m)
    if not meta:
        return None
    params = {
        "location": f"{lat},{lng}",
        "size": f"{size_w}x{size_h}",
        "key": key
    }
    r = requests.get(GOOGLE_STREETVIEW_STATIC, params=params, timeout=60)
    if r.status_code == 200 and r.content:
        return r.content
    return None

# --------------------------
# SerpAPI (reference-only)
# --------------------------
def serpapi_images(query: str, serp_key: str, num: int=4) -> List[Tuple[str, str]]:
    """Return list of (source, thumbnail_url). Reference-only (license unknown)."""
    try:
        url = "https://serpapi.com/search.json"
        params = {"engine": "google", "q": query, "tbm": "isch", "api_key": serp_key}
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        imgs = []
        for item in (data.get("images_results") or [])[:num]:
            link = item.get("original") or item.get("thumbnail") or item.get("link")
            source = item.get("source") or "Google Images via SerpAPI"
            if link:
                imgs.append((source, link))
        return imgs
    except Exception:
        return []

# --------------------------
# Data models
# --------------------------
@dataclass
class Candidate:
    title: str
    source: str
    preview_bytes: bytes
    license_note: str

# --------------------------
# UI — Sidebar
# --------------------------
with st.sidebar:
    st.markdown("### Mode")
    mode = st.radio("",
                    options=("Real Photos", "AI Render"),
                    index=1 if st.session_state.get("mode_ai", True) else 0,
                    label_visibility="collapsed")
    st.session_state["mode_ai"] = (mode == "AI Render")

    st.markdown("### Keys")
    gmaps_key = st.text_input("Google Maps/Places API key", type="password", help="Enable Places API + Street View Static in your Google Cloud project.")
    serp_key  = st.text_input("SerpAPI key (optional)", type="password")
    openai_key = st.text_input("OpenAI API key (for AI Render)", type="password")

    st.markdown("### Output")
    site = st.selectbox("Site style", list(SITE_PROFILES.keys()),
                        index=list(SITE_PROFILES.keys()).index(DEFAULT_SITE))
    base_size = st.selectbox("Render base size (target)", ["1536x1024", "1024x1024", "1024x1536"], index=0)
    out_w, out_h = map(int, base_size.split("x"))
    webp_quality = st.slider("WebP quality", 60, 95, DEFAULT_WEBP_QUALITY)

    if mode == "Real Photos":
        st.markdown("### Sources to use")
        use_places = st.checkbox("Google Places Photos", value=True)
        use_street = st.checkbox("Google Street View", value=True)
        use_serp   = st.checkbox("SerpAPI thumbnails (reference only)", value=False)
        st.markdown("### Street View")
        sv_radius = st.slider("Search radius (meters)", 25, 500, 250, help="Increase if Street View doesn’t appear with the exact place pin.")

    st.markdown("### AI settings")
    lsi_on = st.checkbox("Use LSI expansion (AI Render)", value=False)
    images_per_keyword = st.number_input("Images per keyword (AI Render)", min_value=1, max_value=10, value=1, step=1)

# --------------------------
# UI — Main
# --------------------------
st.title("ImageForge v1.3 — Real Photos + AI Render")

st.caption(
    "Pick a mode in the sidebar. **Real Photos** uses Google Places / Street View (plus optional SerpAPI reference). "
    "**AI Render** uses OpenAI Images with site-aware hints. "
    "Everything exports as WEBP and can be downloaded as a ZIP."
)

keywords_text = st.text_area("Paste keywords (one per line)", height=140, placeholder="Blue Moose Pizza, Vail Colorado\nGolden Retriever getting a bath")
col_a, col_b = st.columns([1,1])

# storage for generated webps to zip
if "generated_zip_items" not in st.session_state:
    st.session_state["generated_zip_items"] = []  # list of (filename, bytes)

def add_to_zip_queue(fname: str, webp_bytes: bytes):
    st.session_state["generated_zip_items"].append((fname, webp_bytes))

def render_downloads():
    items = st.session_state.get("generated_zip_items", [])
    if not items:
        return
    st.success("Done! Download your images below.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, data in items:
            zf.writestr(fname, data)
    buf.seek(0)
    st.download_button("⬇️ Download ZIP", data=buf, file_name=f"imageforge_{int(time.time())}.zip", mime="application/zip")
    st.markdown("### Previews & individual downloads")
    cols = st.columns(3)
    for i, (fname, data) in enumerate(items):
        with cols[i % 3]:
            st.image(data, caption=fname, use_container_width=True)
            st.download_button("Download", data=data, file_name=fname, mime="image/webp", key=f"dl_{i}")

# --------------------------
# Real Photos flow
# --------------------------
def collect_real_photo_candidates(q: str) -> List[Candidate]:
    cands: List[Candidate] = []
    if not gmaps_key:
        st.warning("Please add your Google Maps/Places API key in the sidebar.")
        return cands

    place = google_textsearch_place(q, gmaps_key)
    if not place:
        return cands

    details = google_place_details(place["place_id"], gmaps_key)
    title = details.get("name") or q
    loc = (details.get("geometry") or {}).get("location") or {}
    lat, lng = loc.get("lat"), loc.get("lng")

    # Google Places Photos
    for ph in (details.get("photos") or [])[:8]:
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

    # Street View
    if mode == "Real Photos" and st.session_state.get("use_street", True) and lat and lng:
        try:
            sv = streetview_bytes(lat, lng, gmaps_key, radius_m=st.session_state.get("sv_radius", 250),
                                  size_w=1024, size_h=1024)
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
    if serp_key and st.session_state.get("use_serp", False):
        refs = serpapi_images(q, serp_key, num=4)
        for src, url in refs:
            try:
                r = requests.get(url, timeout=30)
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

# keep user toggle states in session for Real Photos options
if mode == "Real Photos":
    st.session_state["use_serp"]   = 'use_serp' in st.session_state or False
    st.session_state["use_street"] = True
    st.session_state["sv_radius"]  = locals().get("sv_radius", 250)

# --------------------------
# AI Render flow
# --------------------------
def lsi_expand(seed: str, how_many: int) -> List[str]:
    # very lightweight heuristic LSI expansion
    if how_many <= 1:
        return [seed]
    extras = [
        "guide", "tips", "near me", "with kids", "best time",
        "what to wear", "budget", "insider", "local favorites", "map"
    ]
    out = [seed]
    i = 0
    while len(out) < how_many and i < len(extras):
        out.append(f"{seed} {extras[i]}")
        i += 1
    return out

def ai_render_one(keyword: str, site: str, out_w: int, out_h: int, quality: int) -> Optional[Tuple[str, bytes]]:
    if not openai_key:
        st.warning("Please add your OpenAI API key in the sidebar.")
        return None
    prompt = build_prompt(site, keyword)
    try:
        png = openai_generate_image_png(prompt, openai_key)   # always 1024x1024 from API
        webp = to_webp_bytes(png, out_w, out_h, quality)
        return (f"{slugify(keyword)}.webp", webp)
    except Exception as e:
        st.error(f"{keyword}: {e}")
        return None

# --------------------------
# MAIN ACTIONS
# --------------------------
if mode == "AI Render":
    st.session_state["mode_ai"] = True
    # Button must read "Generate Image"
    if col_a.button("Generate Image"):
        kws = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
        if not kws:
            st.warning("Please paste at least one keyword.")
        else:
            # LSI expansion if requested
            final_list: List[str] = []
            for kw in kws:
                if lsi_on and images_per_keyword > 1:
                    final_list.extend(lsi_expand(kw, images_per_keyword))
                else:
                    final_list.append(kw)

            progress = st.progress(0.0)
            for i, kw in enumerate(final_list, start=1):
                r = ai_render_one(kw, site, out_w, out_h, webp_quality)
                if r:
                    add_to_zip_queue(*r)
                progress.progress(i/len(final_list))
            render_downloads()

    if col_b.button("Clear"):
        st.session_state["generated_zip_items"] = []
        st.rerun()

else:
    st.session_state["mode_ai"] = False

    # Real Photos UX: show candidates and a “Create Image” button *on each card*.
    kws = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
    if col_a.button("Collect candidates"):
        st.session_state["realphoto_sets"] = {}
        if not kws:
            st.warning("Please paste at least one keyword.")
        else:
            for kw in kws:
                st.session_state["realphoto_sets"][kw] = collect_real_photo_candidates(kw)

    if col_b.button("Clear"):
        st.session_state["generated_zip_items"] = []
        st.session_state.pop("realphoto_sets", None)
        st.rerun()

    sets = st.session_state.get("realphoto_sets", {})
    for kw, cands in sets.items():
        st.subheader(kw)
        if not cands:
            st.info("No candidates found for this keyword.")
            continue
        cols = st.columns(3)
        for i, c in enumerate(cands):
            with cols[i % 3]:
                st.image(c.preview_bytes, use_container_width=True,
                         caption=f"**[{i}] {c.title}**\n\n{c.license_note}\n\nCredit: {c.source}")
                if st.button("Create Image", key=f"mk_{kw}_{i}"):
                    # Convert chosen candidate to WEBP at target size
                    try:
                        webp = to_webp_bytes(c.preview_bytes, out_w, out_h, webp_quality)
                        add_to_zip_queue(f"{slugify(kw)}.webp", webp)
                        st.success("Added to downloads.")
                    except Exception as e:
                        st.error(f"Failed to process this candidate: {e}")

    render_downloads()
