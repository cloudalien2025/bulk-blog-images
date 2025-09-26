# streamlit_app.py
# =========================================================
# ImageForge v1.3.1  — Real Photos + AI Render + Web Stories
# ---------------------------------------------------------
# - Real Photos: Google Places Photos + Street View Static
# - Optional SerpAPI: reference thumbnails only (non-CC)
# - AI Render: OpenAI Images (gpt-image-1), site-style aware
# - Exports WebP; optional Pinterest (1000x1500) and Web Story (1080x1920)
#
# Notes:
# * Keeps v1.3 behavior; adds only Web Story option and minor resiliency.
# * No 'response_format' is sent to OpenAI to avoid SDK param errors.
# =========================================================

import os, io, re, math, time, json, base64, random, zipfile
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import requests
from PIL import Image, ImageOps, ImageFilter
import streamlit as st

# --------------------------
# Constants / Defaults
# --------------------------
APP_NAME = "ImageForge v1.3.1"
OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

OPENAI_ALLOWED_SIZES = ["1024x1024", "1024x1536", "1536x1024", "auto"]
DEFAULT_OPENAI_SIZE = "1536x1024"

PIN_SIZE = (1000, 1500)      # Pinterest portrait
WEBSTORY_SIZE = (1080, 1920) # Web Story portrait

# --------------------------
# Utilities
# --------------------------
def slugify(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s.lower()).strip()
    s = re.sub(r"\s+", "-", s)
    return s[:120] if len(s) > 120 else s

def season_hint() -> str:
    m = datetime.utcnow().month
    if m in (12,1,2): return "winter"
    if m in (3,4,5):  return "spring"
    if m in (6,7,8):  return "summer"
    return "fall"

def pil_from_bytes(b: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(b))
    return img.convert("RGB")

def save_webp(img: Image.Image, path: str, quality: int = 82):
    img.save(path, "WEBP", quality=int(quality), method=6)

# --- NEW: Web Story export helper (1080×1920) ---
def export_variant_webstory(pil_img: Image.Image, out_dir: str, base_slug: str, webp_quality: int = 82) -> str:
    webstory = ImageOps.fit(
        pil_img, WEBSTORY_SIZE,
        method=Image.LANCZOS,
        centering=(0.5, 0.45)   # keep horizons/signage a bit higher
    )
    out_path = os.path.join(out_dir, f"{base_slug}-webstory.webp")
    save_webp(webstory, out_path, webp_quality)
    return out_path

# Pinterest variant helper
def export_variant_pin(pil_img: Image.Image, out_dir: str, base_slug: str, webp_quality: int = 82) -> str:
    pin = ImageOps.fit(pil_img, PIN_SIZE, method=Image.LANCZOS, centering=(0.5, 0.35))
    out_path = os.path.join(out_dir, f"{base_slug}-pin.webp")
    save_webp(pin, out_path, webp_quality)
    return out_path

# --------------------------
# Google APIs
# --------------------------
def google_places_textsearch(query: str, key: str) -> Dict:
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": key}
    return requests.get(url, params=params, timeout=20).json()

def google_place_details(place_id: str, key: str) -> Dict:
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {"place_id": place_id, "fields": "name,geometry,photos,formatted_address,url", "key": key}
    return requests.get(url, params=params, timeout=20).json()

def google_photo_bytes(photo_ref: str, key: str, max_w: int = 1600) -> Optional[bytes]:
    # Uses redirect to actual CDN image
    url = "https://maps.googleapis.com/maps/api/place/photo"
    params = {"photoreference": photo_ref, "maxwidth": max_w, "key": key}
    r = requests.get(url, params=params, allow_redirects=True, timeout=30)
    if r.status_code == 200:
        return r.content
    return None

def google_streetview_bytes(lat: float, lng: float, key: str, fov: int = 90, radius: int = 100) -> Optional[bytes]:
    # Try metadata to ensure a pano exists nearby
    meta_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    meta_params = {"location": f"{lat},{lng}", "radius": radius, "key": key, "source": "outdoor"}
    m = requests.get(meta_url, params=meta_params, timeout=15).json()
    if m.get("status") != "OK":
        return None

    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "1600x900",
        "location": f"{lat},{lng}",
        "fov": fov,
        "key": key,
        "source": "outdoor"
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 200:
        return r.content
    return None

# --------------------------
# SerpAPI (reference thumbnails only)
# --------------------------
def serpapi_google_images(query: str, key: str, num: int = 3) -> List[Dict]:
    try:
        url = "https://serpapi.com/search.json"
        params = {"engine": "google", "q": query, "tbm": "isch", "ijn": "0", "num": num, "api_key": key}
        d = requests.get(url, params=params, timeout=20).json()
        items = []
        for img in d.get("images_results", [])[:num]:
            items.append({
                "title": img.get("title") or "Google Images (non-CC)",
                "thumb": img.get("thumbnail"),
                "source": "SerpAPI (Google Images, reference only)",
                "license": "Unknown / reference-only",
                "url": img.get("original") or img.get("source")
            })
        return items
    except Exception:
        return []

# --------------------------
# OpenAI (AI Render)
# --------------------------
def openai_generate_image(prompt: str, api_key: str, size: str) -> Optional[Image.Image]:
    """
    Uses OpenAI Images (gpt-image-1).
    Avoids 'response_format' to be SDK-compatible across versions.
    """
    try:
        import openai  # legacy client fallback
        use_legacy = True
    except Exception:
        use_legacy = False

    if use_legacy:
        openai.api_key = api_key
        # OpenAI legacy images endpoint
        resp = openai.Image.create(
            model="gpt-image-1",
            prompt=prompt,
            size=size
        )
        # Try b64_json first; else URL
        data = resp["data"][0]
        b64 = data.get("b64_json")
        if b64:
            return pil_from_bytes(base64.b64decode(b64))
        url = data.get("url")
        if url:
            return pil_from_bytes(requests.get(url, timeout=60).content)
        return None
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=size
        )
        data = resp.data[0]
        # newer SDK usually gives b64_json
        if getattr(data, "b64_json", None):
            return pil_from_bytes(base64.b64decode(data.b64_json))
        if getattr(data, "url", None):
            return pil_from_bytes(requests.get(data.url, timeout=60).content)
        return None

# --------------------------
# LSI helpers (simple/heuristic)
# --------------------------
def lsi_variants(main_kw: str, n: int, site_style: str) -> List[str]:
    if n <= 1: return [main_kw]
    season = season_hint()
    seeds = [
        main_kw,
        f"{main_kw} in {season}",
        f"{main_kw} tips and photos",
        f"best {main_kw.lower()}",
        f"{main_kw} for families",
        f"{main_kw} map and directions",
        f"{main_kw} budget vs luxury",
        f"{main_kw} hidden gems",
        f"{main_kw} rainy day ideas",
        f"{main_kw} during holidays"
    ]
    random.shuffle(seeds)
    return [f"{s} — {site_style} style" for s in seeds[:n]]

def site_style_hint(site: str) -> str:
    # Lightweight style hints; used only to nudge AI prompt
    if "vail" in site:
        return "Vail mountain village, Gore Creek, alpine architecture, gondolas, snowy peaks"
    if "bangkok" in site:
        return "Bangkok urban night vibes, neon markets, temples, tuk-tuks, modern skyline"
    if "boston" in site:
        return "Boston brownstones, waterfront, cobblestone charm, lobster rolls, fall colors"
    if "ipetzo" in site:
        return "Happy pets, natural light, warm lifestyle, playful candid action"
    return "clean editorial travel aesthetic"

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title=APP_NAME, page_icon="🖼️", layout="wide")
st.title(APP_NAME)

with st.sidebar:
    st.markdown("### Mode")
    mode = st.radio("",
        ["Real Photos", "AI Render"],
        key="mode_radio"
    )

    st.markdown("### Keys")
    gmaps_key = st.text_input("Google Maps/Places API key", type="password", help="Required for Real Photos")
    serp_key  = st.text_input("SerpAPI key (optional)", type="password")
    openai_key = st.text_input("OpenAI API key (for AI Render)", type="password")

    st.markdown("### Output")
    webp_quality = st.slider("WebP quality", 50, 95, 82)

    # NEW: Web Story toggle
    make_web_story = st.checkbox("Also make a Web Story image (1080×1920)", value=False)
    make_pin = st.checkbox("Also make a Pinterest image (1000×1500)", value=False)

    st.markdown("### AI settings")
    site_style = st.selectbox("Site style",
                              ["vailvacay.com", "bangkokvacay.com", "bostonvacay.com", "ipetzo.com"],
                              index=0)
    lsi_method = st.selectbox("LSI method", ["Heuristic"], index=0)
    imgs_per_kw = st.selectbox("Images per keyword (LSI expansion)", [1,2,3,4,5,6,7,8,9,10], index=0)

    st.markdown("### Sources to use")
    use_places = st.checkbox("Google Places Photos", value=True, help="Real Photos mode")
    use_street  = st.checkbox("Google Street View", value=True, help="Real Photos mode")
    use_serp    = st.checkbox("SerpAPI thumbnails (reference only)", value=False)

    st.markdown("### Street View")
    sv_radius = st.slider("Search radius (meters)", 50, 400, 150, step=50)

st.caption("Real photos first (Places/Street View + optional SerpAPI refs) or AI Render via OpenAI. Exports WebP (+ optional Pinterest/Web Story).")

# --------------------------
# Main inputs
# --------------------------
keywords = st.text_area("Paste keywords (one per line)", height=120, placeholder="e.g. Things to do in Lionshead Vail\nTavern on the Square, Vail Colorado")

render_size = st.selectbox("Render base size (OpenAI)",
                           OPENAI_ALLOWED_SIZES,
                           index=OPENAI_ALLOWED_SIZES.index(DEFAULT_OPENAI_SIZE))

col_btn1, col_btn2 = st.columns([1,1])
with col_btn1:
    go = st.button("Generate image(s)", type="primary")
with col_btn2:
    clear = st.button("Clear")

if clear:
    st.experimental_rerun()

# --------------------------
# Core engine
# --------------------------
def make_zip(downloads: List[str], label: str = "images.zip"):
    if not downloads: return
    zpath = os.path.join(OUT_DIR, f"{label}")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in downloads:
            zf.write(p, arcname=os.path.basename(p))
    with open(zpath, "rb") as f:
        st.download_button("Download ZIP", f, file_name=os.path.basename(zpath))

def render_ai_variants(kw: str) -> List[Tuple[str, Image.Image]]:
    """Return list of (slug, PIL) for AI renders based on LSI expansion."""
    size = render_size if render_size in ("1024x1024", "1024x1536", "1536x1024") else "1536x1024"
    style_hint = site_style_hint(site_style)
    variants = lsi_variants(kw, int(imgs_per_kw), site_style)
    results = []
    for sub in variants:
        prompt = f"{sub}. Photorealistic, editorial quality. Style: {style_hint}."
        img = openai_generate_image(prompt, openai_key, size)
        if img is None:
            st.error("OpenAI error: OpenAI returned neither b64_json nor url.")
            continue
        results.append((slugify(sub), img))
    return results

def render_real_photo_candidates(kw: str) -> List[Dict]:
    """Collect candidate images for Real Photos mode. Each dict has: title, source, license, bytes."""
    if not gmaps_key:
        st.warning("Google Maps/Places API key is required for Real Photos.")
        return []

    out = []
    # 1) Places search -> details -> photos
    if use_places:
        ts = google_places_textsearch(kw, gmaps_key)
        for res in ts.get("results", [])[:1]:  # top place
            pid = res.get("place_id")
            det = google_place_details(pid, gmaps_key)
            photos = det.get("result", {}).get("photos", []) if det else []
            for ph in photos[:6]:
                b = google_photo_bytes(ph.get("photo_reference"), gmaps_key, max_w=1600)
                if not b: continue
                out.append({
                    "title": det.get("result", {}).get("name", "Google Places Photo"),
                    "source": "Google Places Photo",
                    "license": "Refer to Google Places Photo terms",
                    "bytes": b
                })
            # 2) Street View (near geometry)
            if use_street and det and det.get("result", {}).get("geometry"):
                loc = det["result"]["geometry"]["location"]
                bsv = google_streetview_bytes(loc["lat"], loc["lng"], gmaps_key, fov=90, radius=int(sv_radius))
                if bsv:
                    out.append({
                        "title": f"Google Street View — {det['result'].get('name','')}",
                        "source": "Google Street View",
                        "license": "Refer to Google Street View terms",
                        "bytes": bsv
                    })
            break  # only top place for now

    # 3) SerpAPI thumbnails (reference only)
    if use_serp and serp_key:
        for item in serpapi_google_images(kw, serp_key, num=3):
            try:
                if not item.get("thumb"): 
                    continue
                r = requests.get(item["thumb"], timeout=20)
                if r.status_code == 200:
                    out.append({
                        "title": item["title"],
                        "source": item["source"],
                        "license": item["license"],
                        "bytes": r.content
                    })
            except Exception:
                pass

    return out

downloads: List[str] = []

if go and keywords.strip():
    kw_list = [k.strip() for k in keywords.splitlines() if k.strip()]
    if not kw_list:
        st.stop()

    if mode == "AI Render":
        if not openai_key:
            st.error("OpenAI API key required for AI Render.")
            st.stop()

        for idx, kw in enumerate(kw_list, 1):
            st.subheader(f"{idx}/{len(kw_list)} — {kw}")
            try:
                rendered = render_ai_variants(kw)
            except Exception as e:
                st.error(f"OpenAI error: {e}")
                continue

            cols = st.columns(2)
            for i, (slug, pil_img) in enumerate(rendered):
                with cols[i % 2]:
                    st.image(pil_img, use_container_width=True, caption=slug)
                    base_slug = slugify(kw) if len(rendered)==1 else slug
                    out_path = os.path.join(OUT_DIR, f"{base_slug}.webp")
                    save_webp(pil_img, out_path, webp_quality)
                    downloads.append(out_path)
                    st.download_button("Download", data=open(out_path, "rb"), file_name=os.path.basename(out_path))

                    # Pinterest/Web Story
                    if make_pin:
                        try:
                            pin_path = export_variant_pin(pil_img, OUT_DIR, base_slug, webp_quality)
                            downloads.append(pin_path)
                            st.download_button("Download Pinterest", data=open(pin_path, "rb"), file_name=os.path.basename(pin_path))
                        except Exception as e:
                            st.warning(f"Pinterest export skipped: {e}")

                    if make_web_story:
                        try:
                            ws_path = export_variant_webstory(pil_img, OUT_DIR, base_slug, webp_quality)
                            downloads.append(ws_path)
                            st.download_button("Download Web Story (1080×1920)", data=open(ws_path, "rb"), file_name=os.path.basename(ws_path))
                        except Exception as e:
                            st.warning(f"Web Story export skipped: {e}")

    else:  # Real Photos
        if not gmaps_key:
            st.error("Google Maps/Places API key required for Real Photos.")
            st.stop()

        for idx, kw in enumerate(kw_list, 1):
            st.subheader(f"{idx}/{len(kw_list)} — {kw}")
            cands = render_real_photo_candidates(kw)
            if not cands:
                st.info("No candidates found. Try adjusting the query or enabling Street View / Places.")
                continue

            cols = st.columns(3)
            for i, cand in enumerate(cands):
                with cols[i % 3]:
                    try:
                        pil_img = pil_from_bytes(cand["bytes"])
                    except Exception:
                        continue
                    st.image(pil_img, use_container_width=True,
                             caption=f"{cand['source']} — {cand['title']}")
                    st.caption(f"License: {cand['license']}")

                    # Per-card create & download
                    base_slug = slugify(kw) if i == 0 else f"{slugify(kw)}-{i}"
                    out_path = os.path.join(OUT_DIR, f"{base_slug}.webp")
                    save_webp(pil_img, out_path, webp_quality)
                    downloads.append(out_path)
                    st.download_button("Download", data=open(out_path, "rb"),
                                       file_name=os.path.basename(out_path), key=f"d_{idx}_{i}")

                    if make_pin:
                        try:
                            pin_path = export_variant_pin(pil_img, OUT_DIR, base_slug, webp_quality)
                            downloads.append(pin_path)
                            st.download_button("Download Pinterest", data=open(pin_path, "rb"),
                                               file_name=os.path.basename(pin_path), key=f"p_{idx}_{i}")
                        except Exception as e:
                            st.warning(f"Pinterest export skipped: {e}")

                    if make_web_story:
                        try:
                            ws_path = export_variant_webstory(pil_img, OUT_DIR, base_slug, webp_quality)
                            downloads.append(ws_path)
                            st.download_button("Download Web Story (1080×1920)", data=open(ws_path, "rb"),
                                               file_name=os.path.basename(ws_path), key=f"w_{idx}_{i}")
                        except Exception as e:
                            st.warning(f"Web Story export skipped: {e}")

    # ZIP
    st.markdown("---")
    make_zip(downloads, "images.zip")
    st.success("Done! Download your images above.")
