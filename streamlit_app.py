# ImageForge v1.3.1 — Real Photos + AI Render (Collect Candidates label fixed, OpenAI call patched)
# ------------------------------------------------------------------------------------------------
# Changes vs. original v1.3.1:
# 1) Button label now shows "Collect candidates" in Real Photos mode and "Generate image(s)" in AI Render mode.
# 2) OpenAI image generation uses a robust retry/timeout/HTML-502 handler.
# Everything else remains the same.

import io
import re
import time
import base64
from typing import List, Dict, Optional, Tuple

import requests
from PIL import Image, ImageOps
import streamlit as st

APP_TITLE = "ImageForge v1.3.1"
OPENAI_MODEL = "gpt-image-1"
ALLOWED_SIZES = ["1024x1024", "1024x1536", "1536x1024", "auto"]
DEFAULT_SIZE = "1536x1024"
PINTEREST_SIZE = (1000, 1500)
WEB_STORY_SIZE = (1080, 1920)

# -------------------------
# Utilities
# -------------------------
def sanitize_filename(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\-\_\. ]+", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        text = f"img-{int(time.time())}"
    return text + ".webp"

def pil_to_webp_bytes(img: Image.Image, quality: int = 82) -> bytes:
    out = io.BytesIO()
    img.save(out, format="WEBP", quality=quality, method=6)
    return out.getvalue()

def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img

def center_crop_to(img: Image.Image, target: Tuple[int, int]) -> Image.Image:
    return ImageOps.fit(img, target, method=Image.LANCZOS, centering=(0.5, 0.5))

# -------------------------
# Google Places / Street View
# -------------------------
def google_places_search(api_key: str, query: str, limit: int = 1) -> List[Dict]:
    try:
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {"query": query, "key": api_key}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("results", [])[:max(1, limit)]
    except Exception:
        return []

def google_place_photos(api_key: str, place: Dict, max_count: int = 6) -> List[Tuple[str, str]]:
    out = []
    photos = place.get("photos", [])
    for ph in photos[:max_count]:
        ref = ph.get("photo_reference")
        if not ref:
            continue
        url = "https://maps.googleapis.com/maps/api/place/photo"
        params = {"key": api_key, "photoreference": ref, "maxwidth": 1600}
        photo_url = requests.Request("GET", url, params=params).prepare().url
        attributions = ph.get("html_attributions", [])
        credit = attributions[0] if attributions else "Google Maps contributor"
        out.append((photo_url, credit))
    return out

def street_view_image_urls(api_key: str, lat: float, lng: float, radius_m: int = 150, count: int = 4) -> List[Tuple[str, str]]:
    urls = []
    headings = [0, 90, 180, 270]
    pitches = [0, -5, 10]
    base = "https://maps.googleapis.com/maps/api/streetview"
    for h in headings:
        for p in pitches:
            params = {
                "size": "1600x900",
                "location": f"{lat},{lng}",
                "fov": 90,
                "heading": h,
                "pitch": p,
                "radius": max(1, radius_m),
                "key": api_key,
            }
            url = requests.Request("GET", base, params=params).prepare().url
            urls.append((url, "Google Street View"))
            if len(urls) >= count:
                return urls
    return urls

# -------------------------
# SerpAPI (reference-only thumbnails)
# -------------------------
def serpapi_google_images_thumbs(serpapi_key: str, query: str, max_results: int = 6) -> List[Tuple[str, str]]:
    try:
        url = "https://serpapi.com/search.json"
        params = {"engine": "google_images", "q": query, "num": max_results, "api_key": serpapi_key}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        out = []
        for itm in js.get("images_results", [])[:max_results]:
            thumb = itm.get("thumbnail") or itm.get("original")
            if not thumb:
                continue
            title = itm.get("title", "Google Images (reference)")
            out.append((thumb, f"Google Images via SerpAPI — reference only — {title}"))
        return out
    except Exception:
        return []

# -------------------------
# PATCHED OpenAI call (retry/timeout/502)
# -------------------------
def openai_generate_images(api_key: str, prompt: str, size: str) -> Optional[bytes]:
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}

    max_attempts = 5
    timeout_s = 120
    backoff_base = 1.8

    for attempt in range(1, max_attempts + 1):
        try:
            payload = {"model": OPENAI_MODEL, "prompt": prompt}
            if size and size != "auto":
                payload["size"] = size

            r = requests.post(url, json=payload, headers=headers, timeout=timeout_s)

            ctype = r.headers.get("Content-Type", "")
            looks_html = "text/html" in ctype or (r.text.strip().startswith("<") and "</html>" in r.text.lower())
            if looks_html:
                if attempt < max_attempts:
                    time.sleep(backoff_base ** attempt)
                    continue
                st.error("OpenAI error: Service returned HTML (e.g., 502 gateway). Please try again.")
                return None

            if r.status_code >= 400:
                if r.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts:
                    time.sleep(backoff_base ** attempt)
                    continue
                st.error(f"OpenAI error: {r.status_code} {r.text}")
                return None

            try:
                j = r.json()
            except Exception:
                if attempt < max_attempts:
                    time.sleep(backoff_base ** attempt)
                    continue
                st.error("OpenAI error: Non-JSON response.")
                return None

            data = j.get("data", [])
            if not data:
                if attempt < max_attempts:
                    time.sleep(backoff_base ** attempt)
                    continue
                st.error("OpenAI error: Empty data.")
                return None

            item = data[0]
            if "b64_json" in item and item["b64_json"]:
                return base64.b64decode(item["b64_json"])

            if "url" in item and item["url"]:
                try:
                    img = requests.get(item["url"], timeout=timeout_s)
                    if img.ok and img.content:
                        return img.content
                    if attempt < max_attempts:
                        time.sleep(backoff_base ** attempt)
                        continue
                except requests.RequestException:
                    if attempt < max_attempts:
                        time.sleep(backoff_base ** attempt)
                        continue

            if attempt < max_attempts:
                time.sleep(backoff_base ** attempt)
                continue
            st.error("OpenAI error: Returned neither b64_json nor url.")
            return None

        except requests.Timeout:
            if attempt < max_attempts:
                time.sleep(backoff_base ** attempt)
                continue
            st.error("OpenAI request timed out. Please try again.")
            return None
        except requests.RequestException as e:
            if attempt < max_attempts:
                time.sleep(backoff_base ** attempt)
                continue
            st.error(f"OpenAI request failed: {e}")
            return None
        except Exception as e:
            st.error(f"OpenAI request failed: {e}")
            return None

# -------------------------
# Pipelines
# -------------------------
def fetch_real_photo_for_keyword(
    gmaps_key: str,
    serp_key: Optional[str],
    keyword: str,
    use_places: bool,
    use_street: bool,
    use_serp_ref: bool,
    street_radius: int,
) -> Tuple[Optional[Image.Image], List[Tuple[str, str]]]:
    candidates: List[Tuple[str, str]] = []

    if use_places:
        places = google_places_search(gmaps_key, keyword, limit=1)
        if places:
            p = places[0]
            photos = google_place_photos(gmaps_key, p, max_count=6)
            for url, credit in photos:
                try:
                    r = requests.get(url, timeout=30)
                    if r.ok and r.content:
                        img = Image.open(io.BytesIO(r.content))
                        img = ensure_rgb(img)
                        candidates.append(("Google Places Photo", credit))
                        return img, candidates
                except Exception:
                    continue

            if use_street:
                loc = p.get("geometry", {}).get("location", {})
                lat, lng = loc.get("lat"), loc.get("lng")
                if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
                    sv_urls = street_view_image_urls(gmaps_key, lat, lng, radius_m=street_radius, count=6)
                    for url, credit in sv_urls:
                        try:
                            r = requests.get(url, timeout=30)
                            if r.ok and r.content:
                                img = Image.open(io.BytesIO(r.content))
                                img = ensure_rgb(img)
                                candidates.append(("Google Street View", credit))
                                return img, candidates
                        except Exception:
                            continue

    if serp_key and use_serp_ref:
        refs = serpapi_google_images_thumbs(serp_key, keyword, max_results=4)
        if refs:
            for u, cap in refs:
                candidates.append(("SerpAPI (reference)", cap))

    return None, candidates

def render_ai_for_keyword(openai_key: str, keyword: str, size: str) -> Optional[Image.Image]:
    raw = openai_generate_images(openai_key, keyword, size)
    if not raw:
        return None
    try:
        img = Image.open(io.BytesIO(raw))
        img = ensure_rgb(img)
        return img
    except Exception as e:
        st.error(f"OpenAI returned data, but could not open image: {e}")
        return None

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🖼️", layout="wide")
st.title(APP_TITLE)
st.caption("Real photos first (Places/Street View + optional SerpAPI refs) or AI Render via OpenAI. Exports WebP (+ optional Pinterest/Web Story).")

# Sidebar
st.sidebar.header("Mode")
mode = st.sidebar.radio("", options=["Real Photos", "AI Render"], index=0)

st.sidebar.header("Keys")
gmaps_key = st.sidebar.text_input("Google Maps/Places API key", type="password", help="Required for Places Photos / Street View")
serp_key  = st.sidebar.text_input("SerpAPI key (optional)", type="password", help="Used only for reference thumbnails (never downloaded).")
openai_key = st.sidebar.text_input("OpenAI API key (for AI Render)", type="password")

st.sidebar.header("Output")
webp_quality = int(st.sidebar.slider("WebP quality", 60, 100, 82))

st.sidebar.header("AI settings")
site_style = st.sidebar.selectbox("Site style", ["vailvacay.com", "bangkokvacay.com", "bostonvacay.com", "ipetzo.com"], index=0)
st.sidebar.caption("LSI expansion is used by AI mode.")
lsi_method = st.sidebar.selectbox("LSI method", ["Heuristic"], index=0)
images_per_kw = int(st.sidebar.selectbox("Images per keyword (LSI expansion)", ["1", "2", "3"], index=0))

st.sidebar.header("Sources to use")
use_places = st.sidebar.checkbox("Google Places Photos", value=True)
use_street = st.sidebar.checkbox("Google Street View", value=True)
use_serp_ref = st.sidebar.checkbox("SerpAPI thumbnails (reference only)", value=False)

st.sidebar.header("Street View")
street_radius = int(st.sidebar.slider("Search radius (meters)", 50, 500, 150))

st.sidebar.header("Extras")
make_pinterest = st.sidebar.checkbox("Also make a Pinterest image (1000×1500)", value=False)
make_webstory = st.sidebar.checkbox("Also make a Web Story image (1080×1920)", value=False)

# Main controls
st.subheader("Paste keywords (one per line)")
keywords_txt = st.text_area("", height=110, placeholder="e.g. Things to do in Lionshead Vail\nTavern on the Square, Vail Colorado")

size_choice = st.selectbox("Render base size (OpenAI)", ALLOWED_SIZES, index=ALLOWED_SIZES.index(DEFAULT_SIZE))

# >>> Only change to UI: dynamic label <<<
main_button_label = "Collect candidates" if mode == "Real Photos" else "Generate image(s)"
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    run_btn = st.button(main_button_label, type="primary")
with col_btn2:
    clear_btn = st.button("Clear")

if clear_btn:
    st.experimental_rerun()

if run_btn:
    kws = [k.strip() for k in keywords_txt.splitlines() if k.strip()]
    if not kws:
        st.warning("Please paste at least one keyword.")
    else:
        for idx, kw in enumerate(kws, start=1):
            st.markdown(f"### {idx}/{len(kws)} — {kw}")

            final_img: Optional[Image.Image] = None
            credits: List[Tuple[str, str]] = []

            if mode == "Real Photos":
                if not gmaps_key:
                    st.error("Google Maps/Places API key is required for Real Photos mode.")
                    continue

                final_img, credits = fetch_real_photo_for_keyword(
                    gmaps_key=gmaps_key,
                    serp_key=serp_key if use_serp_ref else None,
                    keyword=kw,
                    use_places=use_places,
                    use_street=use_street,
                    use_serp_ref=use_serp_ref,
                    street_radius=street_radius
                )

            else:  # AI Render
                if not openai_key:
                    st.error("OpenAI API key is required for AI Render mode.")
                    continue

                prompt = kw
                final_img = render_ai_for_keyword(openai_key, prompt, size_choice)

            if not final_img:
                st.error("No image generated or found for this keyword.")
                continue

            webp_bytes = pil_to_webp_bytes(final_img, webp_quality)
            base_name = sanitize_filename(kw)
            st.image(webp_bytes, caption=base_name, use_container_width=True)
            st.download_button("Download", data=webp_bytes, file_name=base_name, mime="image/webp")

            if make_pinterest:
                pin_img = center_crop_to(final_img, PINTEREST_SIZE)
                pin_bytes = pil_to_webp_bytes(pin_img, webp_quality)
                pin_name = sanitize_filename(base_name.replace(".webp", "") + "-pinterest.webp")
                st.download_button("Download Pinterest", data=pin_bytes, file_name=pin_name, mime="image/webp")

            if make_webstory:
                ws_img = center_crop_to(final_img, WEB_STORY_SIZE)
                ws_bytes = pil_to_webp_bytes(ws_img, webp_quality)
                ws_name = sanitize_filename(base_name.replace(".webp", "") + "-webstory.webp")
                st.download_button("Download Web Story", data=ws_bytes, file_name=ws_name, mime="image/webp")

            if credits:
                with st.expander("Provenance / credits"):
                    for cap, cr in credits:
                        st.markdown(f"- **{cap}** — {cr}")
