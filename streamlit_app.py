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


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-")


def pil_to_bytes(img: Image.Image, fmt="PNG", quality=90) -> bytes:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img = img.convert("RGB")
    img.save(buf, format=fmt, quality=quality)
    return buf.getvalue()


def to_webp_bytes(img_bytes: bytes, w: int, h: int, quality: int = 90) -> bytes:
    img = Image.open(io.BytesIO(img_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img = ImageOps.fit(img, (w, h), method=Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality)
    return buf.getvalue()


def download_image_bytes(url: str, timeout: int = 20) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.content
    except Exception:
        return None
    return None


# -----------------------------
# Data classes
# -----------------------------

@dataclass
class RealCandidate:
    title: str
    source: str
    url: str
    preview_bytes: bytes
    meta: dict


# -----------------------------
# Google Places & Street View helpers
# -----------------------------

def places_text_search(query: str, api_key: str) -> Optional[dict]:
    try:
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {"query": query, "key": api_key}
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            data = r.json()
            if data.get("status") in ("OK", "ZERO_RESULTS"):
                return data
    except Exception:
        return None
    return None


def places_details(place_id: str, api_key: str) -> Optional[dict]:
    try:
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {
            "place_id": place_id,
            "key": api_key,
            "fields": "name,formatted_address,photos,geometry,website,url",
        }
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            data = r.json()
            if data.get("status") in ("OK", "ZERO_RESULTS"):
                return data
    except Exception:
        return None
    return None


def places_photo_url(photo_ref: str, api_key: str, maxwidth: int = 1600) -> str:
    return (
        "https://maps.googleapis.com/maps/api/place/photo"
        f"?maxwidth={maxwidth}&photo_reference={photo_ref}&key={api_key}"
    )


def street_view_url(lat: float, lng: float, api_key: str, heading: int = 0, fov: int = 90) -> str:
    return (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size=1600x900&location={lat},{lng}&heading={heading}&fov={fov}&key={api_key}"
    )


def street_view_metadata(lat: float, lng: float, api_key: str, radius: int = 50) -> Optional[dict]:
    try:
        url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        params = {"location": f"{lat},{lng}", "radius": radius, "key": api_key}
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


# -----------------------------
# SerpAPI helpers (optional)
# -----------------------------

def serpapi_image_search(query: str, serp_key: str, num: int = 5) -> List[Tuple[str, dict]]:
    """
    Basic SerpAPI Google Images search.
    Returns list of (image_url, meta).
    """
    out = []
    try:
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google",
            "q": query,
            "tbm": "isch",
            "api_key": serp_key,
            "num": num,
        }
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return out
        data = r.json()
        for img in data.get("images_results", [])[:num]:
            link = img.get("original") or img.get("thumbnail")
            if not link:
                continue
            out.append((link, img))
    except Exception:
        return out
    return out


# -----------------------------
# Core: collect real-photo candidates
# -----------------------------

def collect_real_photo_candidates(
    query: str,
    use_places_flag: bool,
    use_street_flag: bool,
    use_serp_flag: bool,
    sv_radius_m: int,
    gmaps_key: Optional[str],
    serp_key: Optional[str],
) -> List[RealCandidate]:
    cands: List[RealCandidate] = []

    # 1) Google Places Photo(s)
    if use_places_flag and gmaps_key:
        ps = places_text_search(query, gmaps_key)
        if ps and ps.get("results"):
            top = ps["results"][0]
            pid = top.get("place_id")
            if pid:
                det = places_details(pid, gmaps_key)
                if det and det.get("result"):
                    r = det["result"]
                    photos = r.get("photos") or []
                    addr = r.get("formatted_address", "")
                    name = r.get("name", "")
                    loc = (r.get("geometry") or {}).get("location") or {}
                    lat, lng = loc.get("lat"), loc.get("lng")

                    for ph in photos[:12]:
                        ref = ph.get("photo_reference")
                        if not ref:
                            continue
                        url = places_photo_url(ref, gmaps_key, maxwidth=1600)
                        img_bytes = download_image_bytes(url)
                        if not img_bytes:
                            continue
                        title = f"Places Photo — {name}"
                        meta = {
                            "source": "places",
                            "place_id": pid,
                            "name": name,
                            "address": addr,
                            "lat": lat,
                            "lng": lng,
                            "photo_reference": ref,
                        }
                        cands.append(
                            RealCandidate(
                                title=title,
                                source="places",
                                url=url,
                                preview_bytes=img_bytes,
                                meta=meta,
                            )
                        )

    # 2) Street View (near the place centroid)
    if use_street_flag and gmaps_key:
        if not cands and use_places_flag:
            # If we used Places, we might have geometry info there:
            ps = places_text_search(query, gmaps_key)
            if ps and ps.get("results"):
                top = ps["results"][0]
                loc = (top.get("geometry") or {}).get("location") or {}
                lat, lng = loc.get("lat"), loc.get("lng")
                if lat is not None and lng is not None:
                    md = street_view_metadata(lat, lng, gmaps_key, radius=sv_radius_m)
                    if md and md.get("status") == "OK":
                        pano = md.get("pano_id")
                        if pano:
                            loc2 = md.get("location") or {}
                            lat2, lng2 = loc2.get("lat"), loc2.get("lng")
                            # We only build the URL and let the API pick a heading
                            url = (
                                "https://maps.googleapis.com/maps/api/streetview"
                                f"?size=1600x900&pano={pano}&key={gmaps_key}"
                            )
                            img_bytes = download_image_bytes(url)
                            if img_bytes:
                                title = "Street View"
                                meta = {
                                    "source": "streetview",
                                    "pano_id": pano,
                                    "lat": lat2,
                                    "lng": lng2,
                                }
                                cands.append(
                                    RealCandidate(
                                        title=title,
                                        source="streetview",
                                        url=url,
                                        preview_bytes=img_bytes,
                                        meta=meta,
                                    )
                                )

    # 3) SerpAPI (fallback generic images)
    if use_serp_flag and serp_key:
        imgs = serpapi_image_search(query, serp_key, num=5)
        for url, meta in imgs:
            img_bytes = download_image_bytes(url)
            if not img_bytes:
                continue
            title = "SerpAPI Image"
            cands.append(
                RealCandidate(
                    title=title,
                    source="serpapi",
                    url=url,
                    preview_bytes=img_bytes,
                    meta=meta,
                )
            )

    return cands


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title=APP_NAME, page_icon="🖼️", layout="wide")
st.title(APP_NAME)
st.write("Real photos + AI render + batch Excel mode for multiple locations.")

with st.sidebar:
    st.header("API Keys & Settings")
    gmaps_key = st.text_input("Google Maps / Places API key", type="password")
    serp_key = st.text_input("SerpAPI key (optional, for generic images)", type="password")

    st.markdown("---")
    st.subheader("Output Settings")
    quality = st.slider("WEBP quality", 70, 100, 90)
    make_pin = st.checkbox("Also make Pinterest-size variants", value=True)

    st.markdown("---")
    st.subheader("Street View Settings")
    sv_radius_m = st.slider("Street View radius (meters)", 10, 200, 50)

mode = st.radio("Mode", ["Single / Manual", "Batch from Excel"], horizontal=True)

# -----------------------------
# Single / Manual mode
# -----------------------------

if mode == "Single / Manual":
    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        st.subheader("Search & Real Photos")
        query = st.text_input("Business / place / address", "")
        use_places = st.checkbox("Use Google Places Photos", value=True)
        use_street = st.checkbox("Use Google Street View", value=True)
        use_serp = st.checkbox("Use SerpAPI fallbacks", value=False)

        if st.button("Search real photo candidates"):
            if not gmaps_key and (use_places or use_street):
                st.error("Google Maps/Places API key is required for those options.")
            else:
                cands = collect_real_photo_candidates(
                    query,
                    use_places_flag=use_places,
                    use_street_flag=use_street,
                    use_serp_flag=use_serp,
                    sv_radius_m=sv_radius_m,
                    gmaps_key=gmaps_key,
                    serp_key=serp_key,
                )
                if not cands:
                    st.warning("No real-photo candidates found.")
                else:
                    st.success(f"Found {len(cands)} candidates.")
                    for i, c in enumerate(cands, start=1):
                        with st.expander(f"{i}. {c.title} [{c.source}]"):
                            st.image(c.preview_bytes, use_column_width=True)
                            st.code(json.dumps(c.meta, indent=2))

    with col_right:
        st.subheader("AI Render (optional)")
        prompt = st.text_area("AI prompt", "", height=100,
                              placeholder="Describe the scene if you want an AI render...")
        col_a, col_b = st.columns(2)
        size = col_a.selectbox("AI size", ["1024x1024", "1024x576", "576x1024"], index=0)
        num_ai = col_b.slider("How many AI renders?", 1, 4, 1)

        st.info("Hook up to your preferred image generation backend (DALL·E, Stable Diffusion, etc.).")

        if st.button("Generate AI images"):
            st.warning("AI generation not implemented in this template. Add your own DALLE/SD call here.")
            # Example pseudo-code:
            # imgs = call_your_model(prompt, size=size, n=num_ai)
            # for b in imgs: st.image(b)

        st.markdown("---")
        st.subheader("Quick WEBP converter")
        upl = st.file_uploader("Upload any image to convert to 1200x675 WEBP", type=["png", "jpg", "jpeg", "webp"],
                               key="single_webp")
        if upl:
            raw = upl.read()
            webp = to_webp_bytes(raw, OUTPUT_W, OUTPUT_H, quality)
            st.image(webp, caption="Converted preview", use_column_width=True)
            st.download_button("Download WEBP", data=webp, file_name="converted.webp", mime="image/webp")

# -----------------------------
# New: Batch from Excel
# -----------------------------
else:
    st.markdown("### Upload an Excel file with business names")
    st.write(
        "Accepted: `.xlsx` (first row as headers). "
        "Common header names: **name**, **business**, **place**, or choose any column below."
    )

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
        column = st.selectbox(
            "Select the column that contains business names/queries:",
            list(df.columns),
            index=(list(df.columns).index(default_col) if default_col in df.columns else 0),
        )

        # Optional: city/state context to improve disambiguation
        context_hint = st.text_input(
            "Optional location/context to append to each query (e.g., 'Vail, Colorado')",
            "",
        )

        # Batch options
        col1, col2 = st.columns(2)
        images_per_business = col1.number_input(
            "Images per business (saved to ZIP)", 1, 12, 3
        )
        use_street_in_batch = col2.checkbox(
            "Also try Street View if no Places Photo", value=True
        )

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
                        prog.progress(i / total)
                        continue
                    if context_hint:
                        q = f"{q}, {context_hint}"

                    # Collect candidates (this may return multiple Places photos + Street View)
                    cands = collect_real_photo_candidates(
                        q,
                        use_places_flag=True,
                        use_street_flag=use_street_in_batch,
                        use_serp_flag=False,
                        sv_radius_m=sv_radius_m,
                        gmaps_key=gmaps_key,
                        serp_key=None,
                    )

                    # Separate Places and Street View
                    places = [c for c in cands if "Places Photo" in c.title]
                    street = [c for c in cands if "Street View" in c.title]

                    # Build a pool: prefer Places; fall back to Street View
                    pool = places
                    if not pool and use_street_in_batch:
                        pool = street

                    # Limit how many we actually save per business
                    pool = pool[: int(images_per_business)]

                    if not pool:
                        prog.progress(i / total)
                        continue

                    base = slugify(raw) or "image"

                    for j, chosen in enumerate(pool, start=1):
                        try:
                            webp = to_webp_bytes(
                                chosen.preview_bytes, OUTPUT_W, OUTPUT_H, quality
                            )
                            fn = f"{base}_{i}_{j}.webp"
                            zf.writestr(fn, webp)
                            created += 1

                            if make_pin:
                                pin_b = to_webp_bytes(
                                    chosen.preview_bytes, PINTEREST_W, PINTEREST_H, quality
                                )
                                pin_fn = f"{base}_{i}_{j}_pinterest.webp"
                                zf.writestr(pin_fn, pin_b)

                        except Exception as e:
                            st.write(f"⚠️ {raw} (image #{j}): {e}")

                    prog.progress(i / total)

            zip_buf.seek(0)
            st.success(f"Done. Created {created} images.")
            st.download_button(
                "⬇️ Download batch as ZIP",
                data=zip_buf,
                file_name="imageforge_batch.zip",
                mime="application/zip",
            )
