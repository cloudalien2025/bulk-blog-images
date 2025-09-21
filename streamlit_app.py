# ImageForge v2.1 — Real Photos + LSI + Per-Card Generate (with SerpAPI toggle + explicit feedback)
# -----------------------------------------------------------------------------------------------
# - Real photos first: Google Places Photos -> Street View (radius slider)
# - Optional SerpAPI thumbnails (reference only) can be hidden via sidebar
# - Optional OpenAI for LSI keyword expansion (text only)
# - Blog/site style selector (affects filenames/metadata and LSI hints)
# - Images-per-keyword (LSI expansion)
# - Manual pick with clearly labeled thumbnails and a per-card "Make WebP" button
# - Exports 1200x675 WebP (+ optional Pinterest 1000x1500)
# - Robust image validation, explicit success/error messages

import io
import re
import json
from typing import List, Optional, Dict, Tuple

import requests
from PIL import Image, UnidentifiedImageError
import streamlit as st

APP_NAME = "ImageForge v2.1 — Real Photos + LSI + Per-Card Generate"

# Output sizes
MAIN_W, MAIN_H = 1200, 675
PIN_W, PIN_H = 1000, 1500

# Google endpoints
PLACES_TEXTSEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
PLACES_DETAILS_URL    = "https://maps.googleapis.com/maps/api/place/details/json"
PLACES_PHOTO_URL      = "https://maps.googleapis.com/maps/api/place/photo"
SV_IMAGE_URL          = "https://maps.googleapis.com/maps/api/streetview"
SV_META_URL           = "https://maps.googleapis.com/maps/api/streetview/metadata"

# SerpAPI
SERPAPI_IMG_URL = "https://serpapi.com/search.json"

# OpenAI (for LSI only)
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"


# ---------------------------
# Utilities
# ---------------------------

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


# ---------------------------
# Google Places / Street View
# ---------------------------

def places_textsearch(api_key: str, query: str) -> Optional[Dict]:
    params = {"query": query, "key": api_key}
    r = safe_get(PLACES_TEXTSEARCH_URL, params=params)
    if not r: return None
    js = r.json()
    if js.get("status") not in ("OK", "ZERO_RESULTS"): return None
    res = js.get("results", [])
    return res[0] if res else None

def place_details_photos(api_key: str, place_id: str) -> List[Dict]:
    params = {"place_id": place_id, "fields": "photo,geometry,name,url", "key": api_key}
    r = safe_get(PLACES_DETAILS_URL, params=params)
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


# ---------------------------
# SerpAPI (reference only)
# ---------------------------

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


# ---------------------------
# OpenAI LSI (optional)
# ---------------------------

def lsi_terms_openai(api_key: str, keyword: str, site_hint: str, count: int) -> List[str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = (
        f"Generate {count} short LSI keyword variations for the topic below. "
        f"Each item must be under 6 words, no punctuation, natural search phrasing. "
        f"Topic: {keyword}\n"
        f"Site context (use for flavor only): {site_hint}\n"
        f"Return as a JSON array of strings only."
    )
    data = {
        "model": OPENAI_MODEL,
        "messages": [{"role":"user", "content": prompt}],
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
        bits = ["lionshead village", "vail village", "mountain views", "patio dining", "gondola nearby",
                "family friendly", "best breakfast", "happy hour", "gore creek", "apres ski"]
    elif "bangkok" in keyword.lower():
        bits = ["yaowarat", "sukhumvit", "rooftop views", "night market", "street food",
                "best pad thai", "iconic temples", "riverside", "late night", "cozy cafe"]
    elif "boston" in keyword.lower():
        bits = ["seaport district", "north end", "harbor walk", "best lobster roll",
                "back bay", "beacon hill", "family friendly", "harvard square", "brunch spot", "seafood"]
    else:
        bits = ["top rated", "local favorite", "budget friendly", "date night",
                "kid friendly", "near downtown", "with parking", "pet friendly", "open late", "cozy spot"]
    return bits[:max(0, count)]


# ---------------------------
# Candidate model
# ---------------------------

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


# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.caption("Pick a thumbnail and press **Make WebP** on its card. "
           "Only **Places Photos** and **Street View** export. SerpAPI images are **reference-only** (optional to show).")

with st.sidebar:
    st.subheader("Keys")
    gmaps_key = st.text_input("Google Maps/Places API key (required)", type="password")
    serp_key  = st.text_input("SerpAPI key (optional)", type="password")
    openai_key = st.text_input("OpenAI API key (optional, LSI only)", type="password")

    st.subheader("Site style")
    site = st.selectbox("Choose blog/site", ["vailvacay.com", "bangkokvacay.com", "bostonvacay.com", "ipetzo.com"])
    site_hint = {
        "vailvacay.com": "Colorado mountain travel and Vail area picks",
        "bangkokvacay.com": "Bangkok restaurants, nightlife, markets, temples",
        "bostonvacay.com": "Boston neighborhoods, seafood, harbor, North End",
        "ipetzo.com": "General lifestyle and travel"
    }[site]

    st.subheader("Output")
    webp_quality = st.slider("WebP quality", 60, 95, 82)
    make_pin = st.checkbox("Also make a Pinterest image (1000×1500)")

    st.subheader("Street View")
    sv_radius = st.slider("Search radius (meters)", 50, 500, 250)

    st.subheader("Thumbnails")
    include_serpapi = st.checkbox("Include SerpAPI thumbnails (reference-only)", value=False,
                                  help="Uncheck to remove SerpAPI cards completely.")

    st.subheader("LSI & quantity")
    images_per_kw = st.number_input("Images per keyword (LSI expansion)", 1, 12, 1)
    lsi_method = st.selectbox("LSI method", ["Heuristic", "OpenAI (if key provided)"])

keywords_text = st.text_area("Paste keywords (one per line)", height=140,
                             placeholder="Blue Moose Pizza, Vail Colorado\nTavern on the Square, Vail Colorado")
go = st.button("Find thumbnails")

if "export_slots" not in st.session_state:
    st.session_state.export_slots = {}  # {unique_key: bytes}

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
        for p in photos[:6]:
            ref = p.get("photo_reference")

            # small preview
            prev = None
            im_small = fetch_places_photo(gmaps_key, ref, max_w=1024)
            if im_small:
                prev = io.BytesIO()
                crop_resize(im_small, 800, 450).save(prev, "WEBP", quality=80)
                prev = prev.getvalue()

            def make_fetch_full(ref=ref):
                def _f():
                    # try higher width first, then a second-chance lower width
                    for w in (2048, 1600, 1200):
                        im = fetch_places_photo(gmaps_key, ref, max_w=w)
                        if im:
                            return im
                    return None
                return _f

            cands.append(
                Candidate(
                    idx=idx_counter,
                    source="Google Places Photo",
                    title=title,
                    preview=prev,
                    reference_only=False,
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
            prev = io.BytesIO()
            crop_resize(sv_img, 800, 450).save(prev, "WEBP", quality=80)
            prev = prev.getvalue()

            def make_sv_full(lat=lat, lng=lng, r=sv_radius):
                def _f():
                    # attempt bigger first then fallback
                    for size in ((2048,1152),(1600,900),(1280,720)):
                        im = fetch_streetview(gmaps_key, lat, lng, r, size=size)
                        if im:
                            return im
                    return None
                return _f

            cands.append(
                Candidate(
                    idx=idx_counter,
                    source="Google Street View",
                    title=title,
                    preview=prev,
                    reference_only=False,
                    credit="Google Street View",
                    license_str="Refer to Google Street View terms",
                    fetch_full_callable=make_sv_full(),
                )
            )
            idx_counter += 1

    # SerpAPI thumbnails (reference only)
    if include_serpapi and serp_key:
        for m in serpapi_images(kw, serp_key, num=4):
            thumb = None
            if m.get("thumbnail"):
                r = safe_get(m["thumbnail"])
                if r and r.content:
                    thumb = r.content
            cands.append(
                Candidate(
                    idx=idx_counter,
                    source=m["source"],
                    title=m["title"],
                    preview=thumb,
                    reference_only=True,
                    credit="Google Images via SerpAPI — reference only",
                    license_str="Unknown / reference-only",
                    fetch_full_callable=lambda: None,
                )
            )
            idx_counter += 1

    return cands

def render_candidate_card(display_kw: str, site: str, c: Candidate):
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
                      help="SerpAPI/Google Images results are for reference only. Use a Google Places Photo or Street View to export.")
        else:
            if st.button("Make WebP", key=f"make_{display_kw}_{c.idx}"):
                with st.spinner("Fetching and converting..."):
                    img = c.fetch_full()
                    if not img and c.preview:
                        img = valid_img_bytes(c.preview)
                    if not img:
                        st.error("Could not fetch a valid image for export. Try another thumbnail or increase Street View radius.")
                    else:
                        slug = slugify(f"{site}-{display_kw}")
                        webp_main = to_webp_bytes(img, MAIN_W, MAIN_H, webp_quality)
                        st.session_state.export_slots[f"{display_kw}_{c.idx}_main"] = webp_main
                        if make_pin:
                            pin_bytes = to_webp_bytes(img, PIN_W, PIN_H, webp_quality)
                            st.session_state.export_slots[f"{display_kw}_{c.idx}_pin"] = pin_bytes
                        st.success("Done! Download buttons are available on this card.")

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

# Main flow
st.divider()
if go:
    if not gmaps_key:
        st.error("Please provide your Google Maps/Places API key.")
        st.stop()

    raw_lines = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
    if not raw_lines:
        st.warning("Please paste at least one keyword.")
        st.stop()

    # Expand with LSI when needed
    all_tasks: List[Tuple[str, str]] = []  # (display_keyword, base_keyword)
    for base_kw in raw_lines:
        all_tasks.append((base_kw, base_kw))  # always include original
        extra_needed = images_per_kw - 1
        if extra_needed > 0:
            if lsi_method.startswith("OpenAI") and openai_key:
                terms = lsi_terms_openai(openai_key, base_kw, site_hint, extra_needed)
            else:
                terms = lsi_terms_heuristic(base_kw, site, extra_needed)
            for t in terms[:extra_needed]:
                all_tasks.append((f"{base_kw} — {t}", base_kw))

    st.success(f"Loaded {len(all_tasks)} generation slots.")

    for i, (display_kw, base_kw) in enumerate(all_tasks, start=1):
        st.markdown(f"### {i}/{len(all_tasks)} — {display_kw}")
        cands = place_fetch_all_candidates(display_kw)
        if not cands:
            st.warning("No thumbnails found here. Try another radius or slightly different phrasing.")
            continue

        cols = st.columns(3)
        for idx, cand in enumerate(cands):
            with cols[idx % 3]:
                render_candidate_card(display_kw, site, cand)
        st.divider()
