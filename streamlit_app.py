# ImageForge v3.0 — Dual Mode (Real Photos or AI Render) + Per-Card "Generate Image"
# ----------------------------------------------------------------------------------
# Features
# • Mode toggle: Real Photos (Google Places/Street View/SerpAPI) OR AI Render (OpenAI)
# • Clear per-card button: "Generate Image" (replaces Make WebP)
# • Immediate success/error feedback + on-card download buttons
# • LSI expansion (Heuristic or OpenAI) + images-per-keyword
# • Site style selector (affects prompts/filenames)
# • Optional Pinterest image (1000x1500)
# • SerpAPI thumbnails can be hidden
#
# Notes
# • For Real Photos, only Google Places Photo & Street View export (SerpAPI is reference-only).
# • For AI Render, we use OpenAI Images ("gpt-image-1"). We omit `response_format` to avoid
#   the 400 'unknown parameter' issue you've hit. We fetch via returned URL.
# • Provide your own API keys in the sidebar.

import io
import re
import json
from typing import List, Optional, Dict, Tuple

import requests
from PIL import Image, UnidentifiedImageError
import streamlit as st

APP_NAME = "ImageForge v3.0 — Real Photos or AI Render"

# -------------- Output sizes --------------
MAIN_W, MAIN_H = 1200, 675
PIN_W, PIN_H   = 1000, 1500

# -------------- Google endpoints ----------
PLACES_TEXTSEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
PLACES_DETAILS_URL    = "https://maps.googleapis.com/maps/api/place/details/json"
PLACES_PHOTO_URL      = "https://maps.googleapis.com/maps/api/place/photo"
SV_IMAGE_URL          = "https://maps.googleapis.com/maps/api/streetview"
SV_META_URL           = "https://maps.googleapis.com/maps/api/streetview/metadata"

# -------------- SerpAPI -------------------
SERPAPI_IMG_URL = "https://serpapi.com/search.json"

# -------------- OpenAI --------------------
OPENAI_IMAGE_URL = "https://api.openai.com/v1/images/generations"
OPENAI_CHAT_URL  = "https://api.openai.com/v1/chat/completions"
OPENAI_LSI_MODEL = "gpt-4o-mini"


# ================== Utils ==================

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

def valid_img_bytes(b: bytes) -> Optional[Image.Image]:
    try:
        im = Image.open(io.BytesIO(b))
        im.verify()
        im = Image.open(io.BytesIO(b)).convert("RGB")
        return im
    except (UnidentifiedImageError, OSError):
        return None
    except Exception:
        return None

def crop_resize(img: Image.Image, w: int, h: int) -> Image.Image:
    target = w / h
    iw, ih = img.size
    cur = iw / ih
    if cur > target:
        new_w = int(ih * target)
        left = (iw - new_w) // 2
        box = (left, 0, left + new_w, ih)
    else:
        new_h = int(iw / target)
        top = (ih - new_h) // 2
        box = (0, top, iw, top + new_h)
    return img.crop(box).resize((w, h), Image.LANCZOS)

def to_webp_bytes(img: Image.Image, w: int, h: int, quality: int) -> bytes:
    out = io.BytesIO()
    crop_resize(img, w, h).save(out, "WEBP", quality=quality, method=6)
    return out.getvalue()


# ============ Google Places / Street View ============

def places_textsearch(api_key: str, query: str) -> Optional[Dict]:
    r = safe_get(PLACES_TEXTSEARCH_URL, params={"query": query, "key": api_key})
    if not r: return None
    js = r.json()
    if js.get("status") not in ("OK", "ZERO_RESULTS"): return None
    res = js.get("results", [])
    return res[0] if res else None

def place_details_photos(api_key: str, place_id: str) -> List[Dict]:
    r = safe_get(PLACES_DETAILS_URL, params={
        "place_id": place_id, "fields": "photo,geometry,name,url", "key": api_key
    })
    if not r: return []
    js = r.json()
    if js.get("status") != "OK": return []
    return js.get("result", {}).get("photos", []) or []

def fetch_places_photo(api_key: str, photo_ref: str, max_w: int = 1600) -> Optional[Image.Image]:
    try:
        r = requests.get(
            PLACES_PHOTO_URL,
            params={"photoreference": photo_ref, "maxwidth": str(max_w), "key": api_key},
            timeout=30,
            allow_redirects=True,
        )
        if r.status_code == 200:
            return valid_img_bytes(r.content)
    except requests.RequestException:
        pass
    return None

def streetview_meta(api_key: str, lat: float, lng: float, radius_m: int) -> Optional[Dict]:
    r = safe_get(SV_META_URL, params={
        "location": f"{lat},{lng}", "radius": radius_m, "key": api_key, "source": "outdoor"
    })
    if not r: return None
    js = r.json()
    return js if js.get("status") == "OK" else None

def fetch_streetview(api_key: str, lat: float, lng: float, radius_m: int, size=(1600,900)) -> Optional[Image.Image]:
    meta = streetview_meta(api_key, lat, lng, radius_m)
    if not meta: return None
    r = safe_get(SV_IMAGE_URL, params={
        "location": f"{lat},{lng}", "radius": radius_m, "size": f"{size[0]}x{size[1]}",
        "key": api_key, "source": "outdoor"
    })
    if not r: return None
    return valid_img_bytes(r.content)


# ================== SerpAPI ==================

def serpapi_images(query: str, serp_key: str, num: int = 4) -> List[Dict]:
    out = []
    try:
        r = safe_get(SERPAPI_IMG_URL, params={
            "engine": "google", "q": query, "ijn": "0", "tbm": "isch",
            "api_key": serp_key, "num": str(num)
        })
        if not r: return out
        js = r.json()
        for m in js.get("images_results", [])[:num]:
            out.append({
                "title": m.get("title") or "Google Images (reference)",
                "thumbnail": m.get("thumbnail"),
                "original": m.get("original"),
                "source": "SerpAPI (reference-only)",
            })
    except Exception:
        pass
    return out


# ================== OpenAI (LSI + Images) ==================

def lsi_terms_openai(api_key: str, keyword: str, site_hint: str, count: int) -> List[str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = (
        f"Generate {count} short LSI keyword variations for the topic below. "
        f"Each item must be under 6 words, natural search phrasing. "
        f"Topic: {keyword}\nSite context: {site_hint}\n"
        f"Return JSON array of strings only."
    )
    data = {
        "model": OPENAI_LSI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
    }
    try:
        r = requests.post(OPENAI_CHAT_URL, headers=headers, json=data, timeout=30)
        if r.status_code != 200:
            return []
        js = r.json()
        content = js["choices"][0]["message"]["content"]
        arr = json.loads(content)
        return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        return []

def lsi_terms_heuristic(keyword: str, site: str, count: int) -> List[str]:
    if "vail" in keyword.lower():
        bits = ["vail village", "lionshead", "gore creek", "apres ski", "mountain views",
                "patio dining", "happy hour", "family friendly", "near gondola", "breakfast spot"]
    elif "bangkok" in keyword.lower():
        bits = ["sukhumvit", "yaowarat", "rooftop bar", "street food", "night market",
                "riverside", "iconic temples", "late night", "best pad thai", "cozy cafe"]
    elif "boston" in keyword.lower():
        bits = ["north end", "seaport", "harbor walk", "lobster roll", "back bay",
                "beacon hill", "brunch spot", "seafood", "family friendly", "waterfront"]
    else:
        bits = ["top rated", "local favorite", "budget friendly", "date night",
                "kid friendly", "near downtown", "with parking", "pet friendly", "open late", "cozy spot"]
    return bits[:max(0, count)]

def openai_image_url(api_key: str, prompt: str, size: str = "1536x1024") -> Optional[str]:
    # We OMIT response_format to avoid the 'unknown parameter' error you saw.
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "gpt-image-1", "prompt": prompt, "size": size}
    try:
        r = requests.post(OPENAI_IMAGE_URL, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            return None
        js = r.json()
        return js["data"][0].get("url")
    except Exception:
        return None

def build_site_prompt(site: str, keyword: str) -> str:
    base = {
        "vailvacay.com":  "Photorealistic Vail / Lionshead / Gore Creek / alpine village; seasonally correct scenery; no logos; no text.",
        "bangkokvacay.com":"Photorealistic Bangkok city/temples/markets/skyline; warm ambient; no brands; no text.",
        "bostonvacay.com": "Photorealistic Boston neighborhoods/harbor/brick streets; coastal light; no brands; no text.",
        "ipetzo.com":      "Photorealistic lifestyle/travel scenes; neutral; no brands; no text.",
    }.get(site, "Photorealistic editorial travel stock image; no text.")
    # Very light season cue (safe, generic)
    style = "balanced composition; natural light; editorial stock feel."
    return (f"{base} Create an image for: '{keyword}'. Landscape orientation. {style}")

# =============== Candidate model (Real Photos) ===============

class Candidate:
    def __init__(self, idx: int, source: str, title: str, preview: Optional[bytes],
                 reference_only: bool, credit: str, license_str: str,
                 fetch_full_callable):
        self.idx = idx
        self.source = source
        self.title = title
        self.preview = preview
        self.reference_only = reference_only
        self.credit = credit
        self.license_str = license_str
        self.fetch_full = fetch_full_callable


# ========================== UI ===========================

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)

with st.sidebar:
    mode = st.radio("Mode", ["Real Photos", "AI Render"], horizontal=True)

    st.subheader("Keys")
    gmaps_key = st.text_input("Google Maps/Places API key", type="password")
    serp_key  = st.text_input("SerpAPI key (optional)", type="password")
    openai_key = st.text_input("OpenAI API key (for AI/LSI)", type="password")

    st.subheader("Site style")
    site = st.selectbox("Blog/site", ["vailvacay.com", "bangkokvacay.com", "bostonvacay.com", "ipetzo.com"])
    site_hint = {
        "vailvacay.com": "Colorado mountain travel and Vail area picks",
        "bangkokvacay.com": "Bangkok restaurants, nightlife, markets, temples",
        "bostonvacay.com": "Boston neighborhoods, seafood, harbor, North End",
        "ipetzo.com": "General lifestyle and travel"
    }[site]

    st.subheader("Output")
    webp_quality = st.slider("WebP quality", 60, 95, 82)
    make_pin     = st.checkbox("Also make a Pinterest image (1000×1500)")

    if mode == "Real Photos":
        st.subheader("Street View")
        sv_radius = st.slider("Search radius (meters)", 50, 500, 250)
        st.subheader("Thumbnails")
        include_serpapi = st.checkbox("Include SerpAPI thumbnails (reference-only)", value=False)

    st.subheader("LSI & quantity")
    images_per_kw = st.number_input("Images per keyword (LSI expansion)", 1, 12, 1)
    lsi_method = st.selectbox("LSI method", ["Heuristic", "OpenAI (if key provided)"])

keywords_text = st.text_area(
    "Paste keywords (one per line)",
    height=140,
    placeholder="Blue Moose Pizza, Vail Colorado\nBest seafood restaurant in Boston"
)
go = st.button("Find thumbnails" if mode=="Real Photos" else "Prepare prompts")

if "export_slots" not in st.session_state:
    st.session_state.export_slots = {}  # {unique_key: bytes}

# ---------- Real Photos helpers ----------

def place_fetch_all_candidates(kw: str) -> List[Candidate]:
    cands: List[Candidate] = []
    if not gmaps_key:
        return cands

    place = places_textsearch(gmaps_key, kw)
    if not place:
        return cands

    geom = place.get("geometry", {}).get("location", {})
    lat, lng = geom.get("lat"), geom.get("lng")
    pid = place.get("place_id")
    title = place.get("name", kw)

    idx_counter = 0

    # Places Photos
    if pid:
        photos = place_details_photos(gmaps_key, pid)
        for p in photos[:8]:
            ref = p.get("photo_reference")
            prev = None
            im_small = fetch_places_photo(gmaps_key, ref, max_w=1024)
            if im_small:
                prev_io = io.BytesIO()
                crop_resize(im_small, 800, 450).save(prev_io, "WEBP", quality=80)
                prev = prev_io.getvalue()

            def make_fetch_full(ref=ref):
                def _f():
                    for w in (2048, 1600, 1200):
                        im = fetch_places_photo(gmaps_key, ref, max_w=w)
                        if im:
                            return im
                    return None
                return _f

            cands.append(
                Candidate(
                    idx=idx_counter, source="Google Places Photo", title=title,
                    preview=prev, reference_only=False,
                    credit="Google Maps contributor",
                    license_str="Refer to Google Places Photo terms",
                    fetch_full_callable=make_fetch_full(),
                )
            )
            idx_counter += 1

    # Street View
    if lat is not None and lng is not None:
        sv_img = fetch_streetview(gmaps_key, lat, lng, sv_radius, size=(1600, 900))
        prev = None
        if sv_img:
            prev_io = io.BytesIO()
            crop_resize(sv_img, 800, 450).save(prev_io, "WEBP", quality=80)
            prev = prev_io.getvalue()

            def make_sv_full(lat=lat, lng=lng, r=sv_radius):
                def _f():
                    for size in ((2048,1152),(1600,900),(1280,720)):
                        im = fetch_streetview(gmaps_key, lat, lng, r, size=size)
                        if im:
                            return im
                    return None
                return _f

            cands.append(
                Candidate(
                    idx=idx_counter, source="Google Street View", title=title,
                    preview=prev, reference_only=False,
                    credit="Google Street View",
                    license_str="Refer to Google Street View terms",
                    fetch_full_callable=make_sv_full(),
                )
            )
            idx_counter += 1

    # SerpAPI (reference only)
    if 'include_serpapi' in globals() and include_serpapi and serp_key:
        for m in serpapi_images(kw, serp_key, num=6):
            thumb = None
            if m.get("thumbnail"):
                r = safe_get(m["thumbnail"])
                if r and r.content:
                    thumb = r.content
            cands.append(
                Candidate(
                    idx=idx_counter, source=m["source"], title=m["title"],
                    preview=thumb, reference_only=True,
                    credit="Google Images via SerpAPI — reference only",
                    license_str="Unknown / reference-only",
                    fetch_full_callable=lambda: None,
                )
            )
            idx_counter += 1

    return cands

def render_real_card(display_kw: str, site: str, c: Candidate):
    st.markdown(f"**[{c.idx}] {c.source} — {c.title} {'(reference-only)' if c.reference_only else ''}**")
    if c.preview:
        st.image(c.preview, use_container_width=True)
    else:
        st.info("No preview available.")

    st.caption(f"License: {c.license_str}\n\nCredit: {c.credit}")

    col1, col2 = st.columns([1,1])
    with col1:
        if c.reference_only:
            st.button("Why I can't export this", key=f"why_{display_kw}_{c.idx}",
                      help="SerpAPI/Google Images are reference-only. Use a Places Photo or Street View.")
        else:
            if st.button("Generate Image", key=f"gen_{display_kw}_{c.idx}"):
                with st.spinner("Fetching and converting..."):
                    img = c.fetch_full()
                    if not img and c.preview:
                        img = valid_img_bytes(c.preview)
                    if not img:
                        st.error("Could not fetch a valid image. Try a different card or increase Street View radius.")
                    else:
                        slug = slugify(f"{site}-{display_kw}")
                        webp_main = to_webp_bytes(img, MAIN_W, MAIN_H, webp_quality)
                        st.session_state.export_slots[f"{display_kw}_{c.idx}_main"] = webp_main
                        if make_pin:
                            pin_bytes = to_webp_bytes(img, PIN_W, PIN_H, webp_quality)
                            st.session_state.export_slots[f"{display_kw}_{c.idx}_pin"] = pin_bytes
                        st.success("Done! Download buttons are on this card.")

    with col2:
        main_key = f"{display_kw}_{c.idx}_main"
        if main_key in st.session_state.export_slots:
            st.download_button(
                "Download WebP",
                data=st.session_state.export_slots[main_key],
                file_name=f"{slugify(site)}-{slugify(display_kw)}.webp",
                mime="image/webp",
                key=f"dl_{display_kw}_{c.idx}_main"
            )
        pin_key = f"{display_kw}_{c.idx}_pin"
        if pin_key in st.session_state.export_slots:
            st.download_button(
                "Download Pinterest WebP",
                data=st.session_state.export_slots[pin_key],
                file_name=f"{slugify(site)}-{slugify(display_kw)}_pin.webp",
                mime="image/webp",
                key=f"dl_{display_kw}_{c.idx}_pin"
            )

# ---------- AI helpers ----------

def render_ai_slot(display_kw: str, site: str, base_size: str):
    st.markdown(f"**{display_kw}**")
    if st.button("Generate Image", key=f"ai_{display_kw}"):
        if not openai_key:
            st.error("OpenAI API key is required for AI Render.")
            return
        with st.spinner("Rendering via OpenAI…"):
            prompt = build_site_prompt(site, display_kw)
            url = openai_image_url(openai_key, prompt, size=base_size)
            if not url:
                st.error("OpenAI image API failed. Check your key/quota or try again.")
                return
            r = safe_get(url)
            if not r:
                st.error("Failed to download the rendered image URL.")
                return
            img = valid_img_bytes(r.content)
            if not img:
                st.error("Downloaded image was invalid.")
                return
            slug = slugify(f"{site}-{display_kw}")
            webp_main = to_webp_bytes(img, MAIN_W, MAIN_H, webp_quality)
            st.session_state.export_slots[f"{display_kw}_ai_main"] = webp_main
            if make_pin:
                pin = to_webp_bytes(img, PIN_W, PIN_H, webp_quality)
                st.session_state.export_slots[f"{display_kw}_ai_pin"] = pin
            st.success("Done! Download below.")

    main_key = f"{display_kw}_ai_main"
    if main_key in st.session_state.export_slots:
        st.download_button(
            "Download WebP",
            data=st.session_state.export_slots[main_key],
            file_name=f"{slugify(site)}-{slugify(display_kw)}.webp",
            mime="image/webp",
            key=f"dl_{display_kw}_ai_main"
        )
    pin_key = f"{display_kw}_ai_pin"
    if pin_key in st.session_state.export_slots:
        st.download_button(
            "Download Pinterest WebP",
            data=st.session_state.export_slots[pin_key],
            file_name=f"{slugify(site)}-{slugify(display_kw)}_pin.webp",
            mime="image/webp",
            key=f"dl_{display_kw}_ai_pin"
        )

# ================== Main ==================

st.divider()

if go:
    lines = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
    if not lines:
        st.warning("Please paste at least one keyword.")
        st.stop()

    # Build generation list with LSI
    tasks: List[str] = []
    for base_kw in lines:
        tasks.append(base_kw)
        extra = images_per_kw - 1
        if extra > 0:
            if lsi_method.startswith("OpenAI") and openai_key:
                lsis = lsi_terms_openai(openai_key, base_kw, site_hint, extra)
            else:
                lsis = lsi_terms_heuristic(base_kw, site, extra)
            for t in lsis[:extra]:
                tasks.append(f"{base_kw} — {t}")

    st.success(f"Prepared {len(tasks)} generation slot(s).")

    if mode == "Real Photos":
        if not gmaps_key:
            st.error("Google Maps/Places API key required for Real Photos mode.")
            st.stop()

        for i, display_kw in enumerate(tasks, start=1):
            st.markdown(f"### {i}/{len(tasks)} — {display_kw}")
            cands = place_fetch_all_candidates(display_kw)
            if not cands:
                st.warning("No thumbnails found. Increase the Street View radius or try a slightly different phrase.")
                continue

            cols = st.columns(3)
            for idx, cand in enumerate(cands):
                with cols[idx % 3]:
                    render_real_card(display_kw, site, cand)
            st.divider()

    else:  # AI Render
        base_size = "1536x1024"  # good default for 1200×675 crop
        for i, display_kw in enumerate(tasks, start=1):
            st.markdown(f"### {i}/{len(tasks)} — {display_kw}")
            render_ai_slot(display_kw, site, base_size)
            st.divider()
