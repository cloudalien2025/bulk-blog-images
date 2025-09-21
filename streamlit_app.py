# ImageForge v1.1 — Real Photos first
# Streamlit app for collecting real-photo candidates (Google Places Photos + Street View),
# showing labeled thumbnails, letting you pick one, and exporting a 1200×675 WEBP.
#
# Notes
# - Requires: streamlit, requests, pillow
#   pip install streamlit requests pillow
# - Enable "Places API" and "Street View Static API" on your Google Cloud project.
# - SerpAPI is optional and used only for reference thumbnails (not exported).
# - OpenAI is *not* required in this real-photo flow.

import io
import re
import time
import zipfile
from typing import Dict, List, Optional

import requests
import streamlit as st
from PIL import Image


# -----------------------
# Basic UI + state
# -----------------------

st.set_page_config(page_title="ImageForge v1.1 — Real Photos", layout="wide")

if "candidates" not in st.session_state:
    st.session_state.candidates = {}  # kw -> List[Dict]
if "generated" not in st.session_state:
    st.session_state.generated = {}   # kw -> bytes (webp)
if "sv_radius_m" not in st.session_state:
    st.session_state.sv_radius_m = 300


# -----------------------
# Small helpers
# -----------------------

def slugify(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[’'`]", "", t)
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t or "image"

def crop_to_aspect(img: Image.Image, w: int, h: int) -> Image.Image:
    tr = w / h
    W, H = img.size
    r = W / H
    if r > tr:
        newW = int(H * tr); x0 = (W - newW)//2
        box = (x0, 0, x0+newW, H)
    else:
        newH = int(W / tr); y0 = (H - newH)//2
        box = (0, y0, W, y0+newH)
    return img.crop(box)

def _download_image(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> bytes:
    """Download an image (follow redirects), validate with PIL (lightly), and return bytes."""
    r = requests.get(url, params=params or {}, headers=headers or {}, allow_redirects=True, timeout=30)
    r.raise_for_status()
    data = r.content
    # PIL sanity (some thumbnails decode on 2nd try)
    for _ in range(2):
        try:
            Image.open(io.BytesIO(data)).verify()
            return data
        except Exception:
            time.sleep(0.2)
            r = requests.get(r.url, headers=headers or {}, timeout=30)
            data = r.content
    return data


# -----------------------
# Google Places & Street View
# -----------------------

def places_textsearch(api_key: str, query: str) -> Optional[Dict]:
    r = requests.get(
        "https://maps.googleapis.com/maps/api/place/textsearch/json",
        params={"query": query, "key": api_key},
        timeout=30,
    )
    if r.status_code != 200:
        return None
    js = r.json()
    if js.get("status") == "OK" and js.get("results"):
        return js["results"][0]
    return None

def places_details(api_key: str, place_id: str) -> Optional[Dict]:
    r = requests.get(
        "https://maps.googleapis.com/maps/api/place/details/json",
        params={"place_id": place_id, "fields": "name,geometry,photos", "key": api_key},
        timeout=30,
    )
    if r.status_code != 200:
        return None
    js = r.json()
    if js.get("status") == "OK":
        return js.get("result")
    return None

def google_places_candidates(api_key: str, query: str, want_places: bool, want_sv: bool, sv_radius_m: int) -> List[Dict]:
    out: List[Dict] = []
    seed = places_textsearch(api_key, query)
    if not seed:
        return out

    pid = seed.get("place_id")
    det = places_details(api_key, pid) if pid else None
    name = (det or seed).get("name", query)

    # 1) Places Photos
    if want_places:
        photos = (det or {}).get("photos", [])
        for ph in photos[:12]:
            ref = ph.get("photo_reference")
            if not ref:
                continue
            try:
                img = _download_image(
                    "https://maps.googleapis.com/maps/api/place/photo",
                    params={"maxwidth": 1000, "photo_reference": ref, "key": api_key},
                )
                out.append({
                    "title": name,
                    "source": "Google Places Photo",
                    "preview_bytes": img,
                    "usable": True,
                    "meta": {
                        "kind": "places_photo",
                        "place_id": pid,
                        "photo_reference": ref,
                        "license_note": "License: Refer to Google Places Photo terms",
                        "credit": "Google Maps contributor",
                    },
                })
            except Exception:
                continue

    # 2) Street View
    if want_sv:
        loc = ((det or seed).get("geometry") or {}).get("location", {})
        lat, lng = loc.get("lat"), loc.get("lng")
        if lat and lng:
            meta = requests.get(
                "https://maps.googleapis.com/maps/api/streetview/metadata",
                params={"location": f"{lat},{lng}", "radius": sv_radius_m, "key": api_key},
                timeout=20,
            ).json()
            if meta.get("status") == "OK":
                try:
                    sv = _download_image(
                        "https://maps.googleapis.com/maps/api/streetview",
                        params={
                            "size": "1000x650",
                            "location": f"{lat},{lng}",
                            "radius": sv_radius_m,
                            "source": "outdoor",
                            "key": api_key,
                        },
                    )
                    out.append({
                        "title": f"{name} — Street View",
                        "source": "Google Street View",
                        "preview_bytes": sv,
                        "usable": True,
                        "meta": {
                            "kind": "street_view",
                            "lat": lat, "lng": lng, "radius_m": sv_radius_m,
                            "license_note": "License: Refer to Google Street View terms",
                            "credit": "Google Street View",
                        },
                    })
                except Exception:
                    pass
    return out


# -----------------------
# Optional: SerpAPI reference thumbnails (not exported)
# -----------------------

def serpapi_thumbnails(serp_key: str, query: str, limit: int = 6) -> List[Dict]:
    out: List[Dict] = []
    if not serp_key:
        return out
    try:
        # Using Google Images engine
        r = requests.get(
            "https://serpapi.com/search.json",
            params={"engine": "google_images", "q": query, "api_key": serp_key},
            timeout=30,
        )
        if r.status_code != 200:
            return out
        js = r.json()
        for item in (js.get("images_results") or [])[:limit]:
            thumb = item.get("thumbnail") or item.get("original")
            if not thumb:
                continue
            try:
                img = _download_image(thumb, headers={"User-Agent": "Mozilla/5.0"})
            except Exception:
                # If we can't fetch the image bytes reliably, still show as reference (no bytes)
                img = None
            out.append({
                "title": item.get("title") or query,
                "source": "SerpAPI (reference)",
                "preview_bytes": img,  # may be None
                "usable": False,       # block export on purpose
                "meta": {
                    "kind": "serpapi_ref",
                    "link": item.get("link"),
                    "license_note": "Reference only (unknown license). Not exported.",
                    "credit": "Google Images via SerpAPI",
                },
            })
    except Exception:
        return out
    return out


# -----------------------
# Export (Create Image)
# -----------------------

def export_selected(kw: str, cand: Dict, webp_quality: int = 82, pin_long: bool = False):
    if not cand.get("usable"):
        st.warning("This candidate is reference-only and cannot be exported.")
        return

    out_w, out_h = (1200, 675)
    img = Image.open(io.BytesIO(cand["preview_bytes"])).convert("RGB")
    img = crop_to_aspect(img, out_w, out_h).resize((out_w, out_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=webp_quality, method=6)
    data = buf.getvalue()

    st.session_state.generated[kw] = data  # keep the last export for ZIP

    fname = f"{slugify(kw)}.webp"
    st.download_button("⬇️ Download image", data=data, file_name=fname, mime="image/webp")


# -----------------------
# Sidebar
# -----------------------

st.sidebar.markdown("## Mode")
mode = st.sidebar.radio("",
                        ["Real Photos"],
                        index=0,
                        help="This build focuses on real photos (Places Photos + Street View).")

st.sidebar.markdown("## Keys")
g_key = st.sidebar.text_input("Google Maps/Places API key", type="password", help="Required for real photos.")
serp_key = st.sidebar.text_input("SerpAPI key (optional)", type="password",
                                 help="Used only for reference thumbnails from Google Images.")

st.sidebar.markdown("## Sources to use")
use_places = st.sidebar.checkbox("Google Places Photos", value=True)
use_sv = st.sidebar.checkbox("Google Street View", value=True)
use_serp = st.sidebar.checkbox("SerpAPI thumbnails (reference only)", value=False)

st.sidebar.markdown("### Street View")
sv_radius = st.sidebar.slider("Search radius (meters)", 50, 1000, st.session_state.sv_radius_m, step=50,
                              help="Increase if Street View isn’t showing. 300–500m often works well.")
st.session_state.sv_radius_m = sv_radius

st.sidebar.markdown("## Output")
webp_q = st.sidebar.slider("WebP quality", 60, 95, 82)

st.sidebar.markdown("---")
build_zip = st.sidebar.button("Build ZIP of last exports")


# -----------------------
# Main UI
# -----------------------

st.title("ImageForge v1.1 — Real Photos")
st.caption("Real photos first: Google Places Photos → Street View (+ optional SerpAPI reference only). Pick a thumbnail, then **Create Image** to export 1200×675 WEBP.")

keywords_text = st.text_area("Paste keywords (one per line)",
                             height=150,
                             placeholder="Blue Moose Pizza, Vail Colorado\nTavern on the Square, Vail Colorado")

colA, colB = st.columns([1,1])
go = colA.button("Generate candidates")
if colB.button("Clear"):
    st.session_state.candidates.clear()
    st.session_state.generated.clear()
    st.experimental_rerun()


# -----------------------
# Candidate collection
# -----------------------

def collect_for_kw(kw: str) -> List[Dict]:
    cands: List[Dict] = []
    if not g_key:
        st.warning("Enter your Google Maps/Places API key in the sidebar.")
        return cands

    with st.spinner(f"Collecting real-photo candidates for: {kw}"):
        cands.extend(google_places_candidates(
            api_key=g_key,
            query=kw,
            want_places=use_places,
            want_sv=use_sv,
            sv_radius_m=st.session_state.sv_radius_m,
        ))
        if use_serp:
            cands.extend(serpapi_thumbnails(serp_key, kw, limit=6))

    # give each a label index to display
    for i, c in enumerate(cands):
        c["label_idx"] = i
    return cands


if go:
    kws = [ln.strip() for ln in (keywords_text or "").splitlines() if ln.strip()]
    if not kws:
        st.warning("Please paste at least one keyword.")
    else:
        for kw in kws:
            st.session_state.candidates[kw] = collect_for_kw(kw)


# -----------------------
# Show candidates + Create Image
# -----------------------

def show_candidates_for_keyword(kw: str, candidates: List[Dict]):
    st.markdown(f"## {kw}")

    if not candidates:
        st.info("No candidates found. Try widening Street View radius or adjusting the query.")
        return

    pick_key = f"pick_idx_{kw}"
    if pick_key not in st.session_state:
        st.session_state[pick_key] = 0

    cols = st.columns(3)
    for i, c in enumerate(candidates):
        with cols[i % 3]:
            cap = f"**[{i}] {c['source']} — {c['title']}**"
            if c.get("preview_bytes"):
                st.image(c["preview_bytes"], use_container_width=True, caption=cap)
            else:
                st.markdown(cap)
                st.caption("(no preview available)")

            lic = (c.get("meta") or {}).get("license_note") or ("License: Refer to data source terms.")
            cred = (c.get("meta") or {}).get("credit")
            if lic:
                st.caption(lic)
            if cred:
                st.caption(f"Credit: {cred}")

            # radio control to pick this index
            st.radio("Pick",
                     options=[f"Use {i}", "Skip"],
                     index=0 if st.session_state[pick_key] == i else 1,
                     key=f"pick_radio_{kw}_{i}",
                     on_change=lambda k=pick_key, idx=i: st.session_state.__setitem__(k, idx))

            # Action button – Create Image
            if st.button("Create Image", key=f"create_{kw}_{i}"):
                st.session_state[pick_key] = i
                export_selected(kw, c, webp_quality=webp_q)

    # If an image was created for this kw already, show a quick download again
    if kw in st.session_state.generated:
        st.success("Image created. You can download again below.")
        st.download_button("⬇️ Download last image",
                           data=st.session_state.generated[kw],
                           file_name=f"{slugify(kw)}.webp",
                           mime="image/webp")


for kw, cands in st.session_state.candidates.items():
    show_candidates_for_keyword(kw, cands)


# -----------------------
# Build ZIP (last exports)
# -----------------------

if build_zip:
    if not st.session_state.generated:
        st.warning("You haven’t created any images yet.")
    else:
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
            for kw, data in st.session_state.generated.items():
                zf.writestr(f"{slugify(kw)}.webp", data)
        mem.seek(0)
        st.download_button("⬇️ Download ZIP", data=mem,
                           file_name="imageforge_exports.zip",
                           mime="application/zip")
