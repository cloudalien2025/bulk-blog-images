# ImageForge – Real-Photo Mode v1.0
# Streamlit app for real photos of venues/places; AI is not used here.
# Sources: Google Places Photos, Google Street View Static, Openverse (CC).
# Exports: 1200x675 (WebP), optional Pinterest 1000x1500, plus metadata.csv.

import io
import os
import re
import csv
import time
import math
import html
import json
import base64
import zipfile
import textwrap
import requests
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

# ------------------------- UI SETTINGS -------------------------
st.set_page_config(page_title="ImageForge – Real-Photo Mode", layout="wide")

TITLE = "ImageForge v1.0 — Real-Photo Mode"
st.title(TITLE)
st.caption(
    "Real photos first: Google Places Photos → Street View → Openverse (CC). "
    "Lock a specific thumbnail to ‘reference-lock’ it. Exports WebP + metadata.csv."
)

# ------------------------- HELPERS -------------------------
def slugify(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[’'`]", "", t)
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t

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

def to_webp_bytes(img_bytes: bytes, w: int, h: int, quality: int) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = crop_to_aspect(img, w, h).resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=quality, method=6)
    return buf.getvalue()

def strip_html_attrib(html_str: str) -> Tuple[str, Optional[str]]:
    # Google Photos returns html attributions like: <a href="...">Name</a>
    url_match = re.search(r'href="([^"]+)"', html_str or "")
    name = re.sub(r"<[^>]+>", "", html_str or "").strip()
    return name, (url_match.group(1) if url_match else None)

def safe_get(url: str, **kwargs) -> Optional[requests.Response]:
    try:
        return requests.get(url, timeout=30, **kwargs)
    except requests.RequestException:
        return None

# ------------------------- API CALLS -------------------------
def google_places_text_search(query: str, key: str, region: Optional[str]=None) -> List[Dict[str,Any]]:
    base = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": key}
    if region:
        params["region"] = region
    r = safe_get(base, params=params)
    if not r or r.status_code != 200:
        return []
    data = r.json()
    return data.get("results", [])

def google_place_details(place_id: str, key: str) -> Dict[str,Any]:
    base = "https://maps.googleapis.com/maps/api/place/details/json"
    fields = "name,formatted_address,geometry,photo,editorial_summary,url,website"
    params = {"place_id": place_id, "fields": fields, "key": key}
    r = safe_get(base, params=params)
    if not r or r.status_code != 200:
        return {}
    return r.json().get("result", {})

def google_photo_bytes(photo_ref: str, key: str, max_width: int=1600) -> Optional[bytes]:
    # Google redirects to the actual image URL; requests follows by default.
    base = "https://maps.googleapis.com/maps/api/place/photo"
    r = safe_get(base, params={"maxwidth": max_width, "photo_reference": photo_ref, "key": key}, allow_redirects=True)
    if not r or r.status_code != 200:
        return None
    return r.content

def google_street_view_bytes(lat: float, lng: float, key: str, size: str="640x640") -> Optional[bytes]:
    # Standard plan has 640px max. We upscale later. Watermark must remain.
    base = "https://maps.googleapis.com/maps/api/streetview"
    params = {"size": size, "location": f"{lat},{lng}", "key": key}
    r = safe_get(base, params=params)
    if not r or r.status_code != 200:
        return None
    return r.content

def openverse_search(query: str, per_page: int=6) -> List[Dict[str,Any]]:
    url = "https://api.openverse.engineering/v1/images/"
    params = {
        "q": query,
        "page_size": per_page,
        "license": "cc0,by,by-sa",  # permissive licenses
        "license_type": "all",      # include commercial if available
        "mature": "false",
        "format": "json",
    }
    r = safe_get(url, params=params)
    if not r or r.status_code != 200:
        return []
    data = r.json()
    return data.get("results", [])

# ------------------------- CANDIDATE GATHERING -------------------------
def build_candidates_for_keyword(
    kw: str,
    google_key: Optional[str],
    use_places: bool,
    use_streetview: bool,
    use_openverse: bool,
    max_place_results: int,
    max_images_per_kw: int,
) -> List[Dict[str,Any]]:
    """
    Returns a list of candidates:
    {
      'preview_bytes': bytes,
      'full_bytes': bytes,  # same for us
      'source': 'Google Places' | 'Street View' | 'Openverse',
      'title': 'Blue Moose Pizza (Google Places)',
      'credit': 'Photo: Name via Google',
      'license': 'Google Places Terms' | 'CC BY 4.0' | 'CC0' | ...
      'credit_url': 'https://...'
    }
    """
    candidates: List[Dict[str,Any]] = []

    # 1) Google Places Photos
    if google_key and use_places:
        results = google_places_text_search(kw, google_key)[:max_place_results]
        for res in results:
            pid = res.get("place_id")
            det = google_place_details(pid, google_key) if pid else {}
            photos = det.get("photos", []) if det else []
            name = det.get("name") or res.get("name") or kw
            address = det.get("formatted_address") or ""
            lat = det.get("geometry", {}).get("location", {}).get("lat")
            lng = det.get("geometry", {}).get("location", {}).get("lng")

            # Photos first
            for ph in photos:
                if len(candidates) >= max_images_per_kw:
                    break
                ref = ph.get("photo_reference")
                if not ref:
                    continue
                img_bytes = google_photo_bytes(ref, google_key)
                if not img_bytes:
                    continue
                atts = ph.get("html_attributions", [])
                credit_name, credit_url = strip_html_attrib(atts[0]) if atts else ("Google contributor", None)
                candidates.append({
                    "preview_bytes": img_bytes,
                    "full_bytes": img_bytes,
                    "source": "Google Places",
                    "title": f"{name} (Google Places)",
                    "credit": f"Photo: {credit_name} via Google",
                    "license": "Google Maps Platform Terms",
                    "credit_url": credit_url,
                })

            # If no photos and Street View is allowed, add 1 Street View fallback
            if use_streetview and len(candidates) < max_images_per_kw and lat and lng:
                sv_bytes = google_street_view_bytes(lat, lng, google_key)
                if sv_bytes:
                    candidates.append({
                        "preview_bytes": sv_bytes,
                        "full_bytes": sv_bytes,
                        "source": "Street View",
                        "title": f"{name} (Street View)",
                        "credit": "Image © Google Street View (watermark must remain).",
                        "license": "Street View Terms",
                        "credit_url": None,
                    })

            if len(candidates) >= max_images_per_kw:
                break

    # 2) Openverse CC
    if use_openverse and len(candidates) < max_images_per_kw:
        ov = openverse_search(kw, per_page=max_images_per_kw * 2)
        for item in ov:
            if len(candidates) >= max_images_per_kw:
                break
            img_url = item.get("url") or item.get("thumbnail")
            if not img_url:
                continue
            r = safe_get(img_url)
            if not r or r.status_code != 200:
                continue
            creator = item.get("creator") or "Unknown"
            license_ = (item.get("license") or "").upper()
            source_url = item.get("foreign_landing_url") or item.get("url")
            title = item.get("title") or kw
            candidates.append({
                "preview_bytes": r.content,
                "full_bytes": r.content,
                "source": "Openverse",
                "title": f"{title} (Openverse)",
                "credit": f"Photo: {creator} via Openverse",
                "license": f"CC {license_}" if license_ else "CC (see source)",
                "credit_url": source_url,
            })

    return candidates

# ------------------------- UI: SIDEBAR -------------------------
with st.sidebar:
    st.subheader("Keys")
    GOOGLE_KEY = st.text_input("Google Maps/Places API key (required for real photos)", type="password")
    st.caption("Enable Places API + Street View Static API in your Google Cloud project.")

    st.subheader("Output")
    WEBP_W, WEBP_H = 1200, 675
    base_size = st.selectbox("Render base size (source fetch/crop target)", ["1536x1024", "1024x1024", "1024x1536"], index=0)
    quality = st.slider("WebP quality", 60, 95, 82)

    make_pin = st.checkbox("Also make a Pinterest image (1000×1500)", value=False)

    st.subheader("Sources")
    use_places = st.checkbox("Google Places Photos", value=True)
    use_streetview = st.checkbox("Street View fallback", value=True)
    use_openverse = st.checkbox("Openverse (CC only)", value=True)

    st.subheader("Preferences")
    manual_pick = st.selectbox("Thumbnail picking", ["Auto-pick first", "Manual pick (choose below)"], index=1)
    max_place_results = st.slider("Max place results to inspect", 1, 5, 3)
    candidates_per_kw = st.slider("Max candidates per keyword", 1, 8, 4)

st.markdown("### Paste keywords (one per line)")
kw_text = st.text_area(
    "Keywords",
    height=160,
    placeholder="Tavern on the Square, Vail Colorado\nBest seafood restaurant in Boston\nThings to do in Vail in October",
    label_visibility="collapsed"
)
keywords = [ln.strip() for ln in kw_text.splitlines() if ln.strip()]

colA, colB = st.columns([1,1])
go = colA.button("Generate")
clear = colB.button("Clear")
if clear:
    st.experimental_rerun()

# ------------------------- MAIN FLOW -------------------------
if go:
    if not GOOGLE_KEY:
        st.error("Please add your **Google Places API key** in the sidebar.")
        st.stop()
    if not keywords:
        st.warning("Please paste at least one keyword.")
        st.stop()

    st.info("Collecting real-photo candidates. This can take a moment…")
    all_candidates: Dict[str, List[Dict[str,Any]]] = {}
    progress = st.progress(0.0)
    for i, kw in enumerate(keywords, start=1):
        cands = build_candidates_for_keyword(
            kw=kw,
            google_key=GOOGLE_KEY,
            use_places=use_places,
            use_streetview=use_streetview,
            use_openverse=use_openverse,
            max_place_results=max_place_results,
            max_images_per_kw=candidates_per_kw
        )
        all_candidates[kw] = cands
        progress.progress(i / len(keywords))

    st.success("Candidates collected. Review & lock selections (or leave as auto).")

    # SELECTION UI
    selections: Dict[str, int] = {}
    for kw in keywords:
        cands = all_candidates.get(kw, [])
        with st.expander(f"📷 Thumbnails — choose one or leave 'Auto': {kw}", expanded=True):
            if not cands:
                st.warning("No candidates found (even after fallbacks). Try rephrasing the keyword.")
                selections[kw] = -1
            else:
                # Show grid
                n = len(cands)
                cols = st.columns(min(4, n))
                for idx, cand in enumerate(cands):
                    with cols[idx % min(4, n)]:
                        st.image(cand["preview_bytes"], use_container_width=True, caption=f"{idx+1}. {cand['title']}")
                        st.caption(f"{cand['credit']} — {cand['license']}")
                default_label = "Auto (pick 1st)"
                options = [default_label] + [f"Use #{i+1}" for i in range(n)]
                choice = st.selectbox("Selection", options, key=f"sel_{slugify(kw)}")
                selections[kw] = -1 if choice == default_label else (int(choice.split("#")[1]) - 1)

    # FINALIZE
    st.markdown("---")
    finalize = st.button("Create images & ZIP")
    if finalize:
        zip_buf = io.BytesIO()
        zf = zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED)
        meta_rows = [("filename", "keyword", "source", "title", "credit", "license", "credit_url")]

        created = 0
        for kw in keywords:
            cands = all_candidates.get(kw, [])
            if not cands:
                st.error(f"No image for: {kw}")
                continue
            sel_idx = selections.get(kw, -1)
            cand = cands[0] if sel_idx < 0 else cands[sel_idx]

            # Save 1200x675
            try:
                webp = to_webp_bytes(cand["full_bytes"], WEBP_W, WEBP_H, quality)
                fname = f"{slugify(kw)}.webp"
                zf.writestr(fname, webp)
                meta_rows.append((fname, kw, cand["source"], cand["title"], cand["credit"], cand["license"], cand["credit_url"] or ""))

                # Pinterest optional
                if make_pin:
                    pin = to_webp_bytes(cand["full_bytes"], 1000, 1500, quality)
                    pin_name = f"{slugify(kw)}-pinterest.webp"
                    zf.writestr(pin_name, pin)
                    meta_rows.append((pin_name, kw, cand["source"], cand["title"], cand["credit"], cand["license"], cand["credit_url"] or ""))

                created += 1
            except Exception as e:
                st.error(f"{kw}: failed to process — {e}")

        # metadata.csv
        with io.StringIO(newline="") as csv_buf:
            writer = csv.writer(csv_buf)
            writer.writerows(meta_rows)
            zf.writestr("metadata.csv", csv_buf.getvalue())

        zf.close()
        zip_buf.seek(0)

        if created:
            st.success(f"Done! {created} image set(s) created. Download your ZIP below.")
            st.download_button(
                "⬇️ Download ZIP",
                data=zip_buf,
                file_name=f"imageforge_real_photos_{int(time.time())}.zip",
                mime="application/zip"
            )
        else:
            st.warning("No images were created.")

# ------------------------- FOOTER -------------------------
st.markdown("---")
st.caption(
    "Sources prioritized: Google Places Photos → Street View (watermark must remain) → Openverse (CC). "
    "Keep the attribution from **metadata.csv** with each image you publish."
)
