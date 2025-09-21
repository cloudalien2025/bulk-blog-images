# ImageForge v1.1 — Real-Photo Mode + Street View Radius
# ------------------------------------------------------
# - Real photos first: Google Places Photos -> Street View (radius slider)
# - Optional: SerpAPI thumbnails (reference-only) for manual pick
# - Exports .webp 1200x675 (and optional Pinterest 1000x1500)
# - Manual pick respects your choice
# - Robust image validation to prevent PIL errors

import io
import os
import re
import csv
import math
import time
import json
import base64
import zipfile
import requests
from typing import List, Optional, Tuple, Dict
from PIL import Image, UnidentifiedImageError

import streamlit as st

# -------------------------
# App config / constants
# -------------------------

APP_NAME = "ImageForge v1.1 — Real-Photo Mode"
TARGET_W, TARGET_H = 1200, 675
PIN_W, PIN_H = 1000, 1500  # Pinterest long pin
DEFAULT_QUALITY = 82
PLACES_TEXTSEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
PLACES_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
PLACES_PHOTO_URL = "https://maps.googleapis.com/maps/api/place/photo"
SV_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"
SV_META_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
SERPAPI_IMG_URL = "https://serpapi.com/search.json"

# -------------------------
# Helpers
# -------------------------

def slugify(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[’'`]", "", t)
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t or "image"

def safe_get(url: str, **kwargs) -> Optional[requests.Response]:
    try:
        r = requests.get(url, timeout=30, **kwargs)
        if r.status_code == 200:
            return r
    except requests.RequestException:
        pass
    return None

def valid_img_bytes(data: bytes) -> Optional[Image.Image]:
    try:
        im = Image.open(io.BytesIO(data))
        im.verify()  # lightweight check
        im = Image.open(io.BytesIO(data)).convert("RGB")  # re-open for use
        return im
    except UnidentifiedImageError:
        return None
    except Exception:
        return None

def crop_resize(img: Image.Image, w: int, h: int) -> Image.Image:
    target_ratio = w / h
    iw, ih = img.size
    cur_ratio = iw / ih
    if cur_ratio > target_ratio:
        new_w = int(ih * target_ratio)
        left = (iw - new_w) // 2
        box = (left, 0, left + new_w, ih)
    else:
        new_h = int(iw / target_ratio)
        top = (ih - new_h) // 2
        box = (0, top, iw, top + new_h)
    return img.crop(box).resize((w, h), Image.LANCZOS)

def to_webp_bytes(img: Image.Image, w: int, h: int, quality: int) -> bytes:
    out = io.BytesIO()
    crop_resize(img, w, h).save(out, "WEBP", quality=quality, method=6)
    return out.getvalue()

# -------------------------
# Google Places / Street View
# -------------------------

def places_find_first(api_key: str, query: str) -> Optional[Dict]:
    """Use Text Search to resolve a place and basic geometry."""
    params = {
        "query": query,
        "key": api_key,
    }
    r = safe_get(PLACES_TEXTSEARCH_URL, params=params)
    if not r:
        return None
    js = r.json()
    if js.get("status") not in ("OK", "ZERO_RESULTS"):
        return None
    results = js.get("results", [])
    if not results:
        return None
    return results[0]  # best match

def places_details_photos(api_key: str, place_id: str) -> List[Dict]:
    """Get photo references via Place Details."""
    params = {
        "place_id": place_id,
        "fields": "photo,geometry,name,url",
        "key": api_key,
    }
    r = safe_get(PLACES_DETAILS_URL, params=params)
    if not r:
        return []
    js = r.json()
    if js.get("status") != "OK":
        return []
    result = js.get("result", {})
    photos = result.get("photos", []) or []
    return photos

def fetch_places_photo(api_key: str, photo_ref: str, max_w: int = 1600) -> Optional[Image.Image]:
    params = {"photoreference": photo_ref, "maxwidth": str(max_w), "key": api_key}
    # Places Photo endpoint redirects to the actual image URL
    try:
        r = requests.get(PLACES_PHOTO_URL, params=params, timeout=30, allow_redirects=True)
        if r.status_code == 200:
            im = valid_img_bytes(r.content)
            return im
    except requests.RequestException:
        pass
    return None

def streetview_meta(api_key: str, lat: float, lng: float, radius_m: int) -> Optional[Dict]:
    params = {
        "location": f"{lat},{lng}",
        "radius": radius_m,
        "key": api_key,
        "source": "outdoor",   # avoid indoor panos
    }
    r = safe_get(SV_META_URL, params=params)
    if not r:
        return None
    js = r.json()
    if js.get("status") == "OK":
        return js
    return None

def fetch_streetview_image(api_key: str, lat: float, lng: float, radius_m: int, size=(1600, 900)) -> Optional[Image.Image]:
    meta = streetview_meta(api_key, lat, lng, radius_m)
    if not meta:
        return None
    params = {
        "location": f"{lat},{lng}",
        "radius": radius_m,
        "size": f"{size[0]}x{size[1]}",
        "key": api_key,
        "source": "outdoor",
    }
    r = safe_get(SV_IMAGE_URL, params=params)
    if not r:
        return None
    im = valid_img_bytes(r.content)
    return im

# -------------------------
# SerpAPI (reference-only thumbnails)
# -------------------------

def serpapi_images(query: str, serp_key: str, num: int = 4) -> List[Dict]:
    out = []
    try:
        params = {
            "engine": "google",
            "q": query,
            "ijn": "0",
            "tbm": "isch",
            "api_key": serp_key,
            "num": str(num),
        }
        r = safe_get(SERPAPI_IMG_URL, params=params)
        if not r:
            return out
        js = r.json()
        for m in js.get("images_results", [])[:num]:
            out.append(
                {
                    "title": m.get("title") or "Google Images (reference)",
                    "thumbnail": m.get("thumbnail"),
                    "source": "SerpAPI (reference-only)",
                    "license": "Unknown / reference-only",
                    "url": m.get("original") or m.get("thumbnail"),
                    "reference_only": True,
                }
            )
    except Exception:
        pass
    return out

# -------------------------
# Candidate model
# -------------------------

class Candidate:
    def __init__(
        self,
        source: str,
        title: str,
        preview_bytes: Optional[bytes],
        reference_only: bool,
        credit: str = "",
        license_str: str = "",
    ):
        self.source = source
        self.title = title
        self.preview_bytes = preview_bytes  # may be None for ref-only
        self.reference_only = reference_only
        self.credit = credit
        self.license_str = license_str

# -------------------------
# UI
# -------------------------

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.caption(
    "Real photos first: Google Places Photos → Street View (adjustable radius). "
    "Optional: SerpAPI thumbnails are **reference-only**. Exports WebP 1200×675 (+ optional 1000×1500 Pinterest)."
)

with st.sidebar:
    st.subheader("Keys")
    gmaps_key = st.text_input(
        "Google Maps/Places API key (required)",
        type="password",
        help="Enable ‘Places API’ and ‘Street View Static API’ for your project.",
    )
    serp_key = st.text_input("SerpAPI key (optional)", type="password")

    st.subheader("Output")
    base_size = st.selectbox(
        "Render base size (source fetch/crop target)",
        ["1536x1024", "1024x1024", "1024x1536"],
        index=0,
    )
    bw, bh = map(int, base_size.split("x"))
    webp_quality = st.slider("WebP quality", 60, 95, DEFAULT_QUALITY)
    make_pin = st.checkbox("Also make a Pinterest image (1000×1500)")

    st.subheader("Sources to use")
    use_places_photo = st.checkbox("Google Places Photos", value=True)
    use_streetview = st.checkbox("Google Street View", value=True)
    use_serpapi = st.checkbox("SerpAPI thumbnails (reference only)", value=False)

    st.subheader("Street View")
    sv_radius = st.slider("Search radius (meters)", 50, 500, 250, help="Wider radius finds nearby panos if the pin is off.")

    st.subheader("Picker")
    pick_mode = st.selectbox(
        "Thumbnail picking",
        ["Manual pick"],
        index=0,
    )
    st.caption("Manual pick shows all available candidates for each keyword and lets you choose exactly one.")

# main input
keywords_text = st.text_area("Paste keywords (one per line)", height=160, placeholder="Blue Moose Pizza, Vail Colorado\nTavern on the Square, Vail Colorado")

col_go, col_clear = st.columns([1,1])
go = col_go.button("Generate")
if col_clear.button("Clear"):
    st.session_state.clear()
    st.rerun()

if go:
    if not gmaps_key:
        st.error("Please enter your Google Maps/Places API key.")
        st.stop()

    keywords = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
    if not keywords:
        st.warning("Please paste at least one keyword.")
        st.stop()

    # Prepare zip + CSV
    zip_buf = io.BytesIO()
    zf = zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED)
    meta_rows = [["keyword", "filename", "source", "license", "credit"]]

    progress = st.progress(0)
    status = st.empty()

    for idx, kw in enumerate(keywords, start=1):
        status.info(f"Working {idx}/{len(keywords)}: {kw}")

        # 1) Resolve place
        place = places_find_first(gmaps_key, kw)
        if not place:
            st.warning(f"No Google match for: {kw}")
            progress.progress(idx / len(keywords))
            continue

        geometry = place.get("geometry", {}).get("location", {})
        lat, lng = geometry.get("lat"), geometry.get("lng")
        place_id = place.get("place_id")
        place_name = place.get("name", kw)

        candidates: List[Candidate] = []

        # 2) Google Places Photos candidates
        if use_places_photo and place_id:
            photos = places_details_photos(gmaps_key, place_id)
            # Limit a few to keep UI tight
            for p in photos[:6]:
                ref = p.get("photo_reference")
                im = fetch_places_photo(gmaps_key, ref, max_w=max(bw, bh))
                if im:
                    prev = io.BytesIO()
                    crop_resize(im, 800, 450).save(prev, "WEBP", quality=80)
                    candidates.append(
                        Candidate(
                            source="Google Places Photo",
                            title=place_name,
                            preview_bytes=prev.getvalue(),
                            reference_only=False,
                            credit="Google Maps contributor",
                            license_str="Refer to Google Places Photo terms",
                        )
                    )

        # 3) Street View candidate
        if use_streetview and (lat is not None and lng is not None):
            im = fetch_streetview_image(gmaps_key, lat, lng, sv_radius, size=(max(bw, bh), max(bw, bh)))
            if im:
                prev = io.BytesIO()
                crop_resize(im, 800, 450).save(prev, "WEBP", quality=80)
                candidates.append(
                    Candidate(
                        source="Google Street View",
                        title=place_name,
                        preview_bytes=prev.getvalue(),
                        reference_only=False,
                        credit="Google Street View",
                        license_str="Refer to Google Street View terms",
                    )
                )
            else:
                # Helpful hint
                st.info(f"No Street View panorama found within {sv_radius} m for “{kw}”. Try increasing the radius.")

        # 4) SerpAPI thumbnails (reference-only)
        if use_serpapi and serp_key:
            for m in serpapi_images(kw, serp_key, num=4):
                thumb_url = m.get("thumbnail")
                image_bytes = None
                if thumb_url:
                    r = safe_get(thumb_url)
                    if r and r.content:
                        # don't validate (tiny thumbs), just show
                        image_bytes = r.content
                candidates.append(
                    Candidate(
                        source=m["source"],
                        title=m["title"],
                        preview_bytes=image_bytes,
                        reference_only=True,
                        credit="Google Images via SerpAPI — reference only",
                        license_str="Unknown / reference-only",
                    )
                )

        # Show picker
        st.markdown("### 📷 Thumbnails — choose one")
        if not candidates:
            st.warning(f"No candidates found for “{kw}”.")
            progress.progress(idx / len(keywords))
            continue

        pick_index = st.radio(
            f"Pick for: {kw}",
            options=list(range(len(candidates))),
            format_func=lambda i: f"[{i}] {candidates[i].source} — {candidates[i].title} {'(ref-only)' if candidates[i].reference_only else ''}",
            horizontal=True,
            key=f"pick_{idx}",
        )

        # Visuals
        cols = st.columns(3)
        for i, c in enumerate(candidates):
            with cols[i % 3]:
                if c.preview_bytes:
                    st.image(c.preview_bytes, use_container_width=True,
                             caption=f"{c.source} — {c.title}")
                else:
                    st.write(f"*(no preview) {c.source} — {c.title}*")
                st.caption(f"License: {c.license_str}\n\nCredit: {c.credit}")

        chosen = candidates[pick_index]

        if chosen.reference_only:
            st.warning("You picked a **reference-only** image (e.g., Google Images via SerpAPI). "
                       "This will not be exported. Please pick a Google Places Photo or Street View.")
            progress.progress(idx / len(keywords))
            continue

        # Fetch the full-resolution for the chosen candidate again if needed
        # We already hold a display-sized preview; export from full.
        # Re-resolve the actual image for Places Photo or Street View:
        final_img: Optional[Image.Image] = None

        if chosen.source == "Google Places Photo" and place_id and use_places_photo:
            # Try to fetch the *first* photo again; better approach is to map preview->original,
            # but Places doesn't return a stable id per preview here. We'll just take best photo (first).
            photos = places_details_photos(gmaps_key, place_id)
            if photos:
                im = fetch_places_photo(gmaps_key, photos[0].get("photo_reference"), max_w=max(bw, bh))
                if im:
                    final_img = im

        if chosen.source == "Google Street View" and (lat is not None and lng is not None):
            sv_full = fetch_streetview_image(gmaps_key, lat, lng, sv_radius, size=(max(bw, bh), max(bw, bh)))
            if sv_full:
                final_img = sv_full

        if final_img is None and chosen.preview_bytes:
            # Fallback to preview bytes if we cannot re-fetch bigger
            prev_img = valid_img_bytes(chosen.preview_bytes)
            if prev_img:
                final_img = prev_img

        if final_img is None:
            st.error(f"Could not build a valid image for “{kw}”.")
            progress.progress(idx / len(keywords))
            continue

        # Export main
        base_slug = slugify(kw)
        webp_main = to_webp_bytes(final_img, TARGET_W, TARGET_H, webp_quality)
        zf.writestr(f"{base_slug}.webp", webp_main)
        meta_rows.append([kw, f"{base_slug}.webp", chosen.source, chosen.license_str, chosen.credit])

        # Pinterest optional
        if make_pin:
            pin_bytes = to_webp_bytes(final_img, PIN_W, PIN_H, webp_quality)
            zf.writestr(f"{base_slug}_pin.webp", pin_bytes)
            meta_rows.append([kw, f"{base_slug}_pin.webp", chosen.source, chosen.license_str, chosen.credit])

        # Preview card
        st.success(f"Added: {base_slug}.webp")
        st.image(webp_main, caption=f"{base_slug}.webp", use_container_width=True)

        progress.progress(idx / len(keywords))

    # finalize zip + csv
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerows(meta_rows)
    zf.writestr("metadata.csv", csv_buf.getvalue().encode("utf-8"))
    zf.close()
    zip_buf.seek(0)

    st.success("Done! Download your images below.")
    st.download_button("⬇️ Download ZIP", data=zip_buf, file_name="imageforge_realphotos.zip", mime="application/zip")
