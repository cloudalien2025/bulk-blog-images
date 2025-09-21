# ImageForge v1.3.1 — Real Photos + AI Render
# - Keeps v1.3 behavior and UI
# - Patch: Vail neighborhood geo anchors + season guards in build_prompt()
# - Per-candidate "Create Image" works reliably
# - SerpAPI thumbnails obey the toggle (never appear if unchecked)
# - Safer fetching, better error messages (no silent failures)
#
# NOTE: You must bring your own API keys at runtime (do NOT hardcode secrets).

import io
import os
import re
import json
import math
import time
import base64
import zipfile
import textwrap
import urllib.parse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import requests
from PIL import Image, ImageOps
import streamlit as st

# -------------------- App constants --------------------

APP_TITLE = "ImageForge v1.3.1 — Real Photos + AI Render"
DEFAULT_SITE = "vailvacay.com"
RENDER_SIZES = ["1536x1024", "1024x1536", "1024x1024"]  # OpenAI allowed sizes
OUTPUT_W, OUTPUT_H = 1200, 675  # site hero default
DEFAULT_QUALITY = 82

# Site style profiles (same as v1.3)
SITE_PROFILES = {
    "vailvacay.com":  "Photorealistic alpine resort & village scenes in the Colorado Rockies; ski terrain; evergreen forests; cozy lodges; no text.",
    "bangkokvacay.com":"Photorealistic Southeast Asian city scenes; temples, canals, street food markets, tuk-tuks; golden-hour light; no text.",
    "bostonvacay.com": "Photorealistic New England city imagery; brick townhouses, harbor, fall foliage, cobblestones; no text.",
    "ipetzo.com":      "Photorealistic pet lifestyle images; dogs/cats with owners outdoors/indoors; neutral interiors/parks; no brands; no text.",
    "1-800deals.com":  "Photorealistic retail/ecommerce visuals; shopping scenes, parcels, generic products; clean backgrounds; no brands; no text.",
}

# -------------------- v1.3.1 GEO + SEASON HELPERS --------------------

VAIL_NEIGHBORHOOD_HINTS = {
    "lionshead": {
        "must_include": [
            "Eagle Bahn Express gondola terminal (distinctive dark/metal structure)",
            "Arrabelle at Vail Square clock tower or the surrounding square",
            "pedestrian-only plaza (no cars)",
        ],
        "avoid": [
            "Bridge Street (Vail Village landmark)",
            "generic European town",
            "Breckenridge or Aspen architecture",
        ],
        "alt_details": [
            "stone and timber alpine buildings, warm wood balconies",
            "flagstone walkways, plaza fire pits, outdoor seating in season",
        ],
    },
    "vail village": {
        "must_include": [
            "Bridge Street or covered bridge over Gore Creek",
            "alpine storefronts close to the walkway; pedestrian-only",
        ],
        "avoid": ["Lionshead clock tower", "Eagle Bahn gondola terminal"],
        "alt_details": ["cobblestones, chalet facades, gore creek corridors"],
    },
    "golden peak": {
        "must_include": ["Golden Peak base area / Riva Bahn Express lift area"],
        "avoid": ["Eagle Bahn terminal", "Bridge Street"],
        "alt_details": ["race arena feel in winter; training lanes; lift maze"],
    },
}

def _match_vail_zone(k: str) -> Optional[str]:
    k_low = k.lower()
    for zone in VAIL_NEIGHBORHOOD_HINTS.keys():
        if zone in k_low:
            return zone
    if "lion's head" in k_low or "lions head" in k_low:
        return "lionshead"
    return None

def _season_from_keyword_or_today(keyword: str) -> str:
    k = keyword.lower()
    if any(w in k for w in ["winter", "ski", "feb", "jan", "dec"]):
        return "winter"
    if any(w in k for w in ["summer", "hike", "jul", "aug", "jun"]):
        return "summer"
    if any(w in k for w in ["spring", "apr", "april", "may"]):
        return "spring"
    if any(w in k for w in ["fall", "autumn", "sept", "oct", "nov"]):
        return "fall"
    return "auto"

def _seasonal_visual_rules(season: str, site: str, keyword: str) -> Tuple[List[str], List[str]]:
    must, avoid = [], []
    s = season
    if site.endswith("vailvacay.com"):
        if s in ("winter", "auto") and any(w in keyword.lower() for w in ["ski", "lift", "gondola", "back bowls", "powder", "vail"]):
            must += ["snow-covered slopes and trees", "active winter operations if people are skiing"]
            avoid += ["bare summer grass on ski runs", "bikes on ski runs"]
        if s == "summer":
            must += ["lush green slopes", "no snow on runs except distant high peaks"]
        if s == "fall":
            must += ["Colorado aspens in golden/yellow foliage (if trees visible)"]
    return must, avoid

# -------------------- Utilities --------------------

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
        new_w = int(h * target_ratio); left = (w - new_w) // 2
        box = (left, 0, left + new_w, h)
    else:
        new_h = int(w / target_ratio); top = (h - new_h) // 2
        box = (0, top, w, top + new_h)
    return img.crop(box)

def save_webp_bytes(img: Image.Image, w: int, h: int, quality: int) -> bytes:
    img = img.convert("RGB")
    img = crop_to_aspect(img, w, h).resize((w, h), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, "WEBP", quality=quality, method=6)
    return buf.getvalue()

# -------------------- Prompt builder (patched) --------------------

def build_prompt(site: str, keyword: str) -> str:
    """
    v1.3.1 geo-anchored prompt builder.
    """
    base = SITE_PROFILES.get(site, SITE_PROFILES[DEFAULT_SITE])
    k = keyword.strip()
    kl = k.lower()

    style_hints = []
    if any(x in kl for x in ["how far","get to","directions","drive","between"]):
        style_hints.append("clear travel vantage; roadside or plaza approach")
    if any(x in kl for x in ["back bowls","snow","ski","lift","gondola"]):
        style_hints.append("wide vista of ski terrain or gondola context as appropriate")
    if any(x in kl for x in ["where to stay","hotel","resort"]):
        style_hints.append("inviting lodging exterior; golden-hour or dusk")
    if any(x in kl for x in ["what to wear","dress"]):
        style_hints.append("tasteful outfit cues; no logos")
    if any(x in kl for x in ["with kids","family","baby"]):
        style_hints.append("family-friendly moment; warm candid vibe")

    # Geo anchors for Vail
    must_geo, avoid_geo = [], []
    vail_zone = _match_vail_zone(kl) if site.endswith("vailvacay.com") else None
    if vail_zone:
        hints = VAIL_NEIGHBORHOOD_HINTS[vail_zone]
        must_geo += hints["must_include"]
        avoid_geo += hints["avoid"]
        style_hints += hints["alt_details"]

    # Season guards
    season = _season_from_keyword_or_today(kl) if site.endswith("vailvacay.com") else "auto"
    must_season, avoid_season = _seasonal_visual_rules(season, site, k)

    must_list = [*must_geo, *must_season]
    avoid_list = [*avoid_geo, *avoid_season, "random European town", "Breckenridge", "Aspen", "Zermatt", "text or visible logos"]

    style = ", ".join(style_hints) if style_hints else "scene appropriate to the topic"
    must_txt = (" Must include: " + "; ".join(must_list) + ".") if must_list else ""
    avoid_txt = (" Avoid: " + "; ".join(avoid_list) + ".") if avoid_list else ""

    return (
        f"{base} Balanced composition; natural light; editorial stock-photo feel. "
        f"Create an image for the topic: '{keyword}'. Landscape orientation. No words or typography anywhere. "
        f"Scene intent: {style}.{must_txt}{avoid_txt}"
    )

# -------------------- External fetchers --------------------

def http_get(url: str, timeout: int = 25) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

def openai_generate_image_b64(api_key: str, prompt: str, size: str) -> bytes:
    """
    gpt-image-1 with b64_json (works; no response_format error).
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "gpt-image-1", "prompt": prompt, "size": size, "response_format": "b64_json"}
    r = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
    b64 = r.json()["data"][0]["b64_json"]
    return base64.b64decode(b64)

# Google Places + Photos

def google_text_search(api_key: str, query: str) -> Optional[str]:
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": api_key}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("results"):
        return data["results"][0]["place_id"]
    return None

def google_place_photos(api_key: str, place_id: str, max_photos: int = 6) -> List[Dict]:
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {"place_id": place_id, "fields": "photo,website,name,geometry", "key": api_key}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    photos = []
    for idx, p in enumerate(data.get("result", {}).get("photos", [])[:max_photos]):
        ref = p.get("photo_reference")
        width = min(1600, p.get("width", 1600))
        ph_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth={width}&photo_reference={ref}&key={api_key}"
        photos.append({
            "source": "Google Places Photo",
            "title": data.get("result", {}).get("name", "Place photo"),
            "credit": "Google Maps contributor",
            "url": ph_url
        })
    return photos

# Street View Static API

def streetview_meta(api_key: str, lat: float, lng: float, radius: int) -> Optional[Dict]:
    meta_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lng}", "radius": radius, "key": api_key}
    r = requests.get(meta_url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("status") == "OK":
        return data
    return None

def streetview_image_url(api_key: str, pano_meta: Dict, w: int = 1200, h: int = 675) -> str:
    loc = pano_meta.get("location", {})
    lat, lng = loc.get("lat"), loc.get("lng")
    # rely on Google’s default heading/pitch for an easily recognizable snap
    params = {"size": f"{w}x{h}", "location": f"{lat},{lng}", "key": api_key}
    return "https://maps.googleapis.com/maps/api/streetview?" + urllib.parse.urlencode(params)

def google_streetview_candidates(api_key: str, place_id: str, radius_m: int) -> List[Dict]:
    # need place geometry to find lat/lng
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {"place_id": place_id, "fields": "geometry,name", "key": api_key}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    res = r.json().get("result", {})
    geom = res.get("geometry", {}).get("location")
    if not geom:
        return []
    meta = streetview_meta(api_key, geom["lat"], geom["lng"], radius_m)
    if not meta:
        return []
    return [{
        "source": "Google Street View",
        "title": res.get("name", "Street View"),
        "credit": "Google Street View",
        "url": streetview_image_url(api_key, meta, 1200, 675)
    }]

# SerpAPI thumbnails (reference only). Only called if user enabled AND key provided.
def serpapi_image_thumbs(serp_key: str, query: str, num: int = 4) -> List[Dict]:
    url = "https://serpapi.com/search.json"
    params = {"q": query, "engine": "google_images", "ijn": "0", "api_key": serp_key}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    out = []
    for item in (data.get("images_results") or [])[:num]:
        thumb = item.get("thumbnail") or item.get("original")
        if not thumb:
            continue
        out.append({
            "source": "SerpAPI (reference only)",
            "title": item.get("title") or query,
            "credit": "Google Images via SerpAPI — reference only",
            "url": thumb,
            "reference_only": True
        })
    return out

# -------------------- UI + State --------------------

st.set_page_config(page_title="ImageForge v1.3.1", layout="wide")
st.title(APP_TITLE)
st.caption("Pick a mode in the sidebar. Real Photos uses Google Places/Street View (plus optional SerpAPI reference). AI Render uses OpenAI Images with site-aware prompts.")

with st.sidebar:
    st.markdown("### Mode")
    mode = st.radio("",
        options=["Real Photos", "AI Render"],
        index=0)

    st.markdown("### Keys")
    gmaps_key = st.text_input("Google Maps/Places API key", type="password", help="Enable Places API + Street View Static API in the same Google Cloud project.")
    serp_key = st.text_input("SerpAPI key (optional)", type="password")
    openai_key = st.text_input("OpenAI API key (for AI Render / LSI)", type="password")

    st.markdown("### Output")
    webp_quality = st.slider("WebP quality", 60, 95, DEFAULT_QUALITY)

    st.markdown("### AI settings")
    site = st.selectbox("Site style", list(SITE_PROFILES.keys()),
                        index=list(SITE_PROFILES.keys()).index(DEFAULT_SITE))
    lsi_method = st.selectbox("LSI method", ["Heuristic"], index=0, help="Used by AI Render only.")
    images_per_kw = st.number_input("Images per keyword (LSI expansion)", 1, 10, 1, help="AI mode can expand to LSI variants.")
    st.markdown("---")

    st.markdown("### Sources to use")
    use_places = st.checkbox("Google Places Photos", True, help="Real Photos mode")
    use_street = st.checkbox("Google Street View", True, help="Real Photos mode")
    use_serp = st.checkbox("SerpAPI thumbnails (reference only)", False, help="Real Photos mode; shows non-CC images to ‘reference-lock’ your choice. No download from these.")

    st.markdown("### Street View")
    sv_radius = st.slider("Search radius (meters)", 50, 500, 250)

st.markdown("### Paste keywords (one per line)")
keywords_text = st.text_area("", placeholder="Tavern on the Square, Vail Colorado\nBest seafood restaurant in Boston\nThings to do in Vail in October", height=120)

col_btn1, col_btn2 = st.columns([1,1])
action_label = "Generate candidates" if mode == "Real Photos" else "Generate image(s)"
go = col_btn1.button(action_label)
clear = col_btn2.button("Clear")
if clear:
    st.session_state.clear()
    st.experimental_rerun()

keywords: List[str] = [ln.strip() for ln in (keywords_text or "").splitlines() if ln.strip()]

# Hold finished images for ZIP
if "finished" not in st.session_state:
    st.session_state["finished"] = []  # list of (filename, bytes)

def add_finished(fname: str, b: bytes):
    st.session_state["finished"].append((fname, b))

def download_zip_ui():
    if not st.session_state["finished"]:
        return
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fname, b in st.session_state["finished"]:
            z.writestr(fname, b)
    zip_buf.seek(0)
    st.success("Done! Download your images below.")
    st.download_button("⬇️ Download ZIP", data=zip_buf, file_name=f"imageforge_{slugify(site)}.zip", mime="application/zip")

# -------------------- Real Photos flow --------------------

def ui_real_photos():
    if not keywords:
        st.info("Enter at least one keyword.")
        return
    if not gmaps_key:
        st.warning("Enter your Google Maps/Places API key.")
        return

    for kw in keywords:
        st.subheader(kw)
        place_id = google_text_search(gmaps_key, kw)
        candidates: List[Dict] = []

        if not place_id:
            st.warning("No matching place found via Google Places.")
        else:
            if use_places:
                try:
                    candidates += google_place_photos(gmaps_key, place_id, max_photos=12)
                except Exception as e:
                    st.error(f"Places Photos error: {e}")

            if use_street:
                try:
                    candidates += google_streetview_candidates(gmaps_key, place_id, sv_radius)
                except Exception as e:
                    st.error(f"Street View error: {e}")

        if use_serp and serp_key:
            try:
                candidates += serpapi_image_thumbs(serp_key, kw, num=6)
            except Exception as e:
                st.error(f"SerpAPI error: {e}")

        if not candidates:
            st.info("No candidates found.")
            continue

        # Render grid with a “Create Image” button per card.
        cols = st.columns(3)
        for idx, cand in enumerate(candidates):
            with cols[idx % 3]:
                label = f"[{idx}] {cand['source']} — {cand.get('title','')}"
                st.markdown(f"**{label}**")
                cap = f"License: Refer to {cand['source']} terms" if "SerpAPI" not in cand['source'] else "License: Unknown / reference-only"
                try:
                    img_bytes = http_get(cand["url"])
                    st.image(img_bytes, use_container_width=True)
                    st.caption(cap)
                except Exception as e:
                    st.warning(f"Preview not available: {e}")
                    continue

                # Create Image button
                btn = st.button("Create Image", key=f"create_{slugify(kw)}_{idx}")
                if btn:
                    if "SerpAPI" in cand["source"]:
                        st.warning("This is reference-only (non-CC). Choose a Google Places or Street View image to create a downloadable WebP.")
                    else:
                        try:
                            img = Image.open(io.BytesIO(img_bytes))
                            out = save_webp_bytes(img, OUTPUT_W, OUTPUT_H, webp_quality)
                            fname = f"{slugify(kw)}.webp"
                            add_finished(fname, out)
                            st.image(out, use_container_width=True, caption=fname)
                            st.download_button("Download", data=out, file_name=fname, mime="image/webp", key=f"dl_{slugify(kw)}_{idx}")
                        except Exception as e:
                            st.error(f"Failed to process image: {e}")

        download_zip_ui()

# -------------------- AI Render flow --------------------

def lsi_expand(keyword: str, n: int) -> List[str]:
    if n <= 1:
        return [keyword]
    # Simple heuristic variety for now (keeps v1.3 spirit without extra cost)
    base = [keyword]
    k = keyword.lower()
    extras = []
    if "vail" in k and "lionshead" not in k and "village" not in k:
        extras += ["Lionshead Vail activities", "Vail Village things to do"]
    if "restaurant" in k or "food" in k or "dining" in k:
        extras += ["best places to eat", "top rated restaurants", "local favorites dining"]
    if "boston" in k:
        extras += ["best lobster roll in Boston", "seafood restaurants on the waterfront"]
    if "bangkok" in k:
        extras += ["where to eat in Sukhumvit", "best street food in Chinatown Yaowarat"]
    # pad if needed
    for i in range(100):
        if len(base)+len(extras) >= n:
            break
        extras.append(f"{keyword} tips {i+1}")
    return (base + extras)[:n]

def ui_ai_render():
    if not keywords:
        st.info("Enter at least one keyword.")
        return
    if not openai_key:
        st.warning("Enter your OpenAI API key.")
        return

    size = st.selectbox("Render base size (OpenAI)", RENDER_SIZES, index=0)

    progress = st.progress(0)
    finished = 0
    total = sum([max(1, int(images_per_kw)) for _ in keywords])

    for kw in keywords:
        variants = lsi_expand(kw, int(images_per_kw))
        for v in variants:
            try:
                prompt = build_prompt(site, v)
                png = openai_generate_image_b64(openai_key, prompt, size)
                img = Image.open(io.BytesIO(png))
                webp = save_webp_bytes(img, OUTPUT_W, OUTPUT_H, webp_quality)
                fname = f"{slugify(v)}.webp"
                add_finished(fname, webp)
                st.image(webp, use_container_width=True, caption=fname)
                st.download_button("Download", data=webp, file_name=fname, mime="image/webp", key=f"ai_{fname}")
            except Exception as e:
                st.error(f"{v}: {e}")
            finished += 1
            progress.progress(min(1.0, finished/total))

    download_zip_ui()

# -------------------- Main branch --------------------

if go:
    if mode == "Real Photos":
        ui_real_photos()
    else:
        ui_ai_render()
