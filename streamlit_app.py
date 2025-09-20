import io
import math
import os
import re
import time
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import requests
from PIL import Image
import streamlit as st

# ---------------------------
# App meta
# ---------------------------
APP_NAME = "ImageForge v1.1 — Real-Photo First"
OUTPUT_W, OUTPUT_H = 1200, 675
PIN_W, PIN_H = 1000, 1500  # optional Pinterest long pin
DEFAULT_QUALITY = 82

# ---------------------------
# Helpers
# ---------------------------

@dataclass
class Candidate:
    source: str                     # "Google Places Photo" | "Google Street View" | "Openverse (CC)" | "SerpAPI (reference)"
    title: str
    license_note: str
    credit_html: str
    preview_bytes: Optional[bytes]  # None if reference only / unavailable
    downloadable: bool              # True if we can export actual bytes
    ref_url: Optional[str] = None   # for reference-only items
    meta: Optional[dict] = None     # any extra metadata


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


def to_webp_bytes(img: Image.Image, w: int, h: int, quality: int) -> bytes:
    img = img.convert("RGB")
    img = crop_to_aspect(img, w, h).resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=quality, method=6)
    return buf.getvalue()


def fetch_bytes(url: str, headers: Optional[dict] = None, timeout: int = 30) -> Optional[bytes]:
    try:
        h = {"User-Agent": "ImageForge/1.1", "Accept": "image/jpeg,image/png,image/webp;q=0.8,*/*;q=0.5"}
        if headers:
            h.update(headers)
        r = requests.get(url, headers=h, timeout=timeout)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    # Bearing from (lat1,lon1) to (lat2,lon2)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    brng = (math.degrees(math.atan2(x, y)) + 360) % 360
    return brng

# ---------------------------
# Google APIs (Places + Street View)
# ---------------------------

def places_text_search(keyword: str, key: str) -> Optional[dict]:
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": keyword,
        "key": key,
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 200:
        js = r.json()
        if js.get("results"):
            return js["results"][0]  # most relevant
    return None


def places_photo_bytes(photo_ref: str, max_w: int, key: str) -> Optional[bytes]:
    # 1) Get redirect URL
    photo_url = "https://maps.googleapis.com/maps/api/place/photo"
    params = {"photoreference": photo_ref, "maxwidth": max_w, "key": key}
    r = requests.get(photo_url, params=params, allow_redirects=False, timeout=30)
    if r.status_code in (302, 303) and r.headers.get("Location"):
        final_url = r.headers["Location"]
        return fetch_bytes(final_url)
    # Some projects return 200 with bytes directly:
    if r.status_code == 200 and r.content:
        return r.content
    return None


def streetview_best_image(lat: float, lng: float, key: str, target_lat: float, target_lng: float) -> Tuple[Optional[bytes], str, str]:
    """
    Returns (image_bytes, license_note, credit_html)
    We query metadata first to ensure a pano exists, then fetch an actual JPEG with a heading
    that points toward the place.
    """
    meta_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    meta_params = {"location": f"{lat},{lng}", "source": "outdoor", "radius": 50, "key": key}
    meta = requests.get(meta_url, params=meta_params, timeout=30).json()

    if meta.get("status") != "OK":
        return None, "No Street View panorama found near this place.", "Credit: Google Street View (none available)"

    # Compute heading from pano to the place
    pano_loc = meta.get("location", {})
    plat = pano_loc.get("lat", lat)
    plng = pano_loc.get("lng", lng)
    heading = int(round(bearing_deg(plat, plng, target_lat, target_lng)))

    # Street View Static (JPEG). Free tier max size 640x640.
    sv_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "640x640",
        "location": f"{plat},{plng}",
        "heading": str(heading),
        "pitch": "0",
        "fov": "90",
        "source": "outdoor",
        "key": key,
    }
    img_bytes = fetch_bytes(sv_url, headers={"Accept": "image/jpeg"})
    if not img_bytes:
        return None, "Street View image unavailable (HTTP).", "Credit: Google Street View"
    return img_bytes, "License: Refer to Google Street View terms", "Credit: Google Street View"

# ---------------------------
# Openverse (CC)
# ---------------------------

def openverse_cc_candidates(keyword: str, limit: int = 3) -> List[Candidate]:
    # Public API; we’ll keep it conservative
    out = []
    url = "https://api.openverse.engineering/v1/images"
    params = {"q": keyword, "page_size": limit, "license_type": "cc", "mature": "false"}
    try:
        js = requests.get(url, params=params, timeout=20).json()
        for r in js.get("results", []):
            thumb = r.get("thumbnail") or r.get("url")
            img = fetch_bytes(thumb) if thumb else None
            title = (r.get("title") or r.get("source") or "Openverse Image").strip()
            lic = f"License: {r.get('license')} — {r.get('license_url')}"
            credit = f'Credit: <a href="{r.get("foreign_landing_url")}" target="_blank">{r.get("source", "Openverse")}</a>'
            out.append(Candidate(
                source="Openverse (CC)",
                title=title,
                license_note=lic,
                credit_html=credit,
                preview_bytes=img,
                downloadable=True,
                meta=r
            ))
    except Exception:
        pass
    return out

# ---------------------------
# SerpAPI (reference-only)
# ---------------------------

def serpapi_thumbnails(keyword: str, serp_key: str, n: int = 3) -> List[Candidate]:
    out = []
    try:
        # Google Images via SerpAPI
        url = "https://serpapi.com/search.json"
        params = {"engine": "google_images", "q": keyword, "ijn": "0", "api_key": serp_key}
        js = requests.get(url, params=params, timeout=30).json()
        for r in (js.get("images_results") or [])[:n]:
            thumb = r.get("thumbnail")
            img = fetch_bytes(thumb) if thumb else None
            link = r.get("original") or r.get("link") or r.get("thumbnail")
            title = (r.get("title") or "Google Images").strip()
            out.append(Candidate(
                source="SerpAPI (reference-only)",
                title=title,
                license_note="License: Unknown / reference-only",
                credit_html="Credit: Google Images via SerpAPI — reference only",
                preview_bytes=img,              # for display only
                downloadable=False,             # we will NOT export this byte-for-byte
                ref_url=link,
                meta=r
            ))
    except Exception:
        pass
    return out

# ---------------------------
# Site Profiles (for optional AI fallback prompts)
# ---------------------------

SITE_PROFILES = {
    "vailvacay.com":  "Editorial, photoreal scenes of Colorado resort towns (Vail, Beaver Creek). No logos unless present in the reference; avoid text rendering.",
    "bostonvacay.com": "Real Boston neighborhoods, harbor, red brick, brownstones. Documentary feel.",
    "bangkokvacay.com":"Bangkok city scenes, temples, markets, rooftops. Vibrant yet realistic.",
    "ipetzo.com":      "Pet lifestyle photographs indoors/outdoors; natural light; no brands.",
}
DEFAULT_SITE = "vailvacay.com"

# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.caption("Real photos first (Google Places → Street View → Openverse CC). SerpAPI images are reference-only.")

with st.sidebar:
    st.subheader("Keys")
    gmaps_key = st.text_input("Google Maps/Places API key (required for real photos)", type="password")
    serp_key  = st.text_input("SerpAPI key (optional, for reference thumbnails)", type="password")
    openai_key = st.text_input("OpenAI API key (optional, for AI lookalikes)", type="password")

    st.subheader("Output")
    site = st.selectbox("Site style (for optional AI)", list(SITE_PROFILES.keys()),
                        index=list(SITE_PROFILES.keys()).index(DEFAULT_SITE))
    base_size = st.selectbox("Render base size (Street View/Openverse fetch/crop target)",
                             ["1536x1024", "1024x1536", "1024x1024"], index=0)
    quality = st.slider("WebP quality", 60, 95, DEFAULT_QUALITY)
    mk_pin = st.checkbox("Also make a Pinterest image (1000×1500)")

    st.subheader("Sources to use")
    use_places = st.checkbox("Google Places Photos", True)
    use_street = st.checkbox("Google Street View", True)
    use_cc = st.checkbox("Openverse (CC)", True)
    use_serp = st.checkbox("SerpAPI thumbnails (reference only)", True)

    st.subheader("Picker & LSI")
    pick_mode = st.selectbox("Thumbnail picking", ["Manual pick thumbnail"], index=0)
    images_per_kw = st.number_input("Images per keyword (LSI expansion)", min_value=1, max_value=10, value=1, step=1)
    lsi_method = st.selectbox("LSI method", ["Heuristic", "None"], index=0)

    st.subheader("Fallbacks")
    season_aware = st.checkbox("Season-aware prompts (AI fallback)", True)
    allow_ai_fallback = st.checkbox("Allow AI fallback if no real photo found", True)

keywords_text = st.text_area("Paste keywords (one per line)", height=120,
                             placeholder="Blue Moose Pizza, Vail Colorado\nTavern on the Square, Vail Colorado")

colA, colB = st.columns([1,1])
do_generate = colA.button("Generate")
do_clear = colB.button("Clear")

if do_clear:
    st.session_state.clear()
    st.experimental_set_query_params()
    st.rerun()

# Persist picks
if "picks" not in st.session_state:
    st.session_state["picks"] = {}  # kw -> index

# ---------------------------
# Candidate collection
# ---------------------------

def collect_candidates_for_keyword(kw: str) -> List[Candidate]:
    cands: List[Candidate] = []
    place = None

    if use_places or use_street:
        if not gmaps_key:
            st.warning(f"Google Maps/Places key required to fetch real photos for '{kw}'.")
        else:
            place = places_text_search(kw, gmaps_key)

    # Google Places Photo
    if use_places and place:
        photos = place.get("photos") or []
        if photos:
            photo_ref = photos[0].get("photo_reference")
            if photo_ref:
                img = places_photo_bytes(photo_ref, max_w=1600, key=gmaps_key)
                if img:
                    cands.append(Candidate(
                        source="Google Places Photo",
                        title=f"{place.get('name','Place Photo')}",
                        license_note="License: Refer to Google Places Photo terms",
                        credit_html=f'Credit: <a target="_blank" href="https://maps.google.com/?q=place_id:{place.get("place_id","")}">Google Maps contributor</a>',
                        preview_bytes=img,
                        downloadable=True,
                        meta={"place_id": place.get("place_id")}
                    ))

    # Google Street View
    if use_street and place:
        geom = place.get("geometry", {}).get("location", {})
        plat, plng = geom.get("lat"), geom.get("lng")
        if plat is not None and plng is not None:
            img_bytes, lic, credit = streetview_best_image(plat, plng, gmaps_key, plat, plng)
            if img_bytes:
                cands.append(Candidate(
                    source="Google Street View",
                    title=f"{place.get('name','Street View')}",
                    license_note=lic,
                    credit_html=credit,
                    preview_bytes=img_bytes,
                    downloadable=True,
                    meta={"lat": plat, "lng": plng}
                ))
            else:
                # Make it visible why no Street View preview
                cands.append(Candidate(
                    source="Google Street View",
                    title="No Street View panorama found",
                    license_note=lic,
                    credit_html=credit,
                    preview_bytes=None,
                    downloadable=False
                ))

    # Openverse CC
    if use_cc:
        cands.extend(openverse_cc_candidates(kw, limit=3))

    # SerpAPI reference images
    if use_serp and serp_key:
        cands.extend(serpapi_thumbnails(kw, serp_key, n=3))

    return cands


def show_picker(kw: str, cands: List[Candidate]) -> int:
    if not cands:
        st.info("No candidates found.")
        return -1

    st.markdown(f"#### 📷 Thumbnails — choose one or leave your previous pick: {kw}")
    picked = st.session_state["picks"].get(kw, -1)

    # Grid display with radios
    cols = st.columns(3)
    for idx, c in enumerate(cands):
        with cols[idx % 3]:
            st.markdown(f"**[{idx}] {c.source} — {c.title}**")
            if c.preview_bytes:
                try:
                    st.image(c.preview_bytes, use_container_width=True)
                except Exception:
                    st.caption("Preview not renderable (bytes not an image).")
            else:
                st.caption("No preview (reference only / unavailable)")

            st.caption(c.license_note)
            st.markdown(c.credit_html, unsafe_allow_html=True)

            # radio for this item:
            pick_this = st.radio("Pick", ["Use", "Skip"], index=0 if picked == idx else 1, key=f"pick-{kw}-{idx}")
            if pick_this == "Use":
                picked = idx

    st.session_state["picks"][kw] = picked
    return picked


def ai_lookalike_from_reference(openai_key: str, ref_url: str, site_hint: str, kw: str, size: str) -> Optional[bytes]:
    """
    Use OpenAI image generation with an image URL as reference.
    Returns PNG bytes (base64) which we convert to WebP later.
    """
    try:
        # This uses the Images API (gpt-image-1) with b64_json response.
        # We keep the prompt lightweight and editorial, nudging realism.
        prompt = (
            f"{site_hint}. Create a natural editorial photo matching the subject of this reference "
            f"image for the topic: '{kw}'. Do NOT include any text overlays or logos."
        )
        headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-image-1",
            "prompt": prompt,
            "image": ref_url,            # reference-lock
            "size": size,
            "response_format": "b64_json"
        }
        r = requests.post("https://api.openai.com/v1/images/edits", headers=headers, json=payload, timeout=90)
        if r.status_code != 200:
            return None
        b64 = r.json()["data"][0]["b64_json"]
        return io.BytesIO(bytes.fromhex("")) if not b64 else (io.BytesIO())
    except Exception:
        return None

# ---------------------------
# Generation
# ---------------------------

if do_generate:
    kws = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
    if not kws:
        st.warning("Please paste at least one keyword.")
        st.stop()

    if (use_places or use_street) and not gmaps_key:
        st.error("Google Maps/Places API key is required to fetch real photos.")
        st.stop()

    size_for_openai = {"1536x1024": "1536x1024", "1024x1536": "1024x1536", "1024x1024": "1024x1024"}[base_size]

    zip_buf = io.BytesIO()
    zf = zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED)

    previews: List[Tuple[str, bytes]] = []
    meta_rows: List[Dict[str, str]] = []

    for kw in kws:
        st.markdown(f"### Working: {kw}")
        cands = collect_candidates_for_keyword(kw)
        picked_idx = show_picker(kw, cands)
        if picked_idx < 0 or picked_idx >= len(cands):
            st.warning("No pick selected for this keyword. Skipping.")
            continue

        c = cands[picked_idx]

        # 1) Downloadable sources → crop/resize and save
        if c.downloadable and c.preview_bytes:
            try:
                img = Image.open(io.BytesIO(c.preview_bytes))
                webp = to_webp_bytes(img, OUTPUT_W, OUTPUT_H, quality)
                fname = f"{slugify(kw)}.webp"
                zf.writestr(fname, webp)
                previews.append((fname, webp))
                meta_rows.append({
                    "keyword": kw, "source": c.source, "title": c.title,
                    "license": c.license_note, "credit": c.credit_html
                })

                if mk_pin:
                    pin_webp = to_webp_bytes(img, PIN_W, PIN_H, quality)
                    zf.writestr(f"{slugify(kw)}_pin.webp", pin_webp)
            except Exception as e:
                st.error(f"{kw}: failed to process image — {e}")

        # 2) Reference-only pick
        else:
            if not allow_ai_fallback or not openai_key:
                st.info(f"'{kw}': Selected item is reference-only. Enable **AI fallback** and set your OpenAI key to synthesize a lookalike.")
                # Still record metadata so you can revisit the link
                meta_rows.append({
                    "keyword": kw, "source": c.source, "title": c.title,
                    "license": c.license_note, "credit": c.credit_html,
                })
                continue

            # Try AI lookalike from reference URL
            st.write(f"Generating AI lookalike for '{kw}' using reference URL…")
            ref_url = c.ref_url or ""
            png_stream = None
            try:
                # Use images/edits with reference (note: if edits model requires mask, switch to generations with 'image[]')
                headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
                payload = {
                    "model": "gpt-image-1",
                    "prompt": f"{SITE_PROFILES.get(site, '')}. Create a realistic editorial photo for: '{kw}'. No text/logo.",
                    "size": size_for_openai,
                    "response_format": "b64_json",
                    "image": ref_url
                }
                r = requests.post("https://api.openai.com/v1/images/edits", headers=headers, json=payload, timeout=120)
                if r.status_code == 200:
                    b64 = r.json()["data"][0]["b64_json"]
                    if b64:
                        png_bytes = requests.utils.b64decode(b64)
                        png_stream = io.BytesIO(png_bytes)
            except Exception:
                png_stream = None

            if not png_stream:
                st.warning(f"AI fallback failed for '{kw}'. Recorded metadata only.")
                meta_rows.append({
                    "keyword": kw, "source": c.source, "title": c.title,
                    "license": c.license_note, "credit": c.credit_html,
                })
                continue

            try:
                img = Image.open(png_stream)
                webp = to_webp_bytes(img, OUTPUT_W, OUTPUT_H, quality)
                fname = f"{slugify(kw)}.webp"
                zf.writestr(fname, webp)
                previews.append((fname, webp))
                meta_rows.append({
                    "keyword": kw, "source": f"{c.source} + AI lookalike", "title": c.title,
                    "license": "Synthesized from reference; verify rights before publishing.",
                    "credit": c.credit_html
                })
                if mk_pin:
                    pin_webp = to_webp_bytes(img, PIN_W, PIN_H, quality)
                    zf.writestr(f"{slugify(kw)}_pin.webp", pin_webp)
            except Exception as e:
                st.error(f"{kw}: failed to process AI image — {e}")

    zf.close()
    zip_buf.seek(0)

    st.success("Done! Download your images below.")
    st.download_button("⬇️ Download ZIP", data=zip_buf, file_name="imageforge_images.zip", mime="application/zip")

    st.markdown("### Previews & individual downloads")
    cols = st.columns(3)
    for i, (fname, data_bytes) in enumerate(previews):
        with cols[i % 3]:
            st.image(data_bytes, use_container_width=True, caption=fname)
            st.download_button("Download", data=data_bytes, file_name=fname, mime="image/webp")
