# ImageForge v1.2 — Real Photos + AI Render
# pip install streamlit requests pillow

import io
import re
import time
import zipfile
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image


# -----------------------
# App Config
# -----------------------
st.set_page_config(page_title="ImageForge v1.2 — Real Photos + AI Render", layout="wide")

if "candidates" not in st.session_state:
    st.session_state.candidates = {}  # kw -> List[Dict]
if "generated" not in st.session_state:
    st.session_state.generated = {}   # kw -> List[Tuple[str, bytes]]  (many images per kw in AI mode)
if "sv_radius_m" not in st.session_state:
    st.session_state.sv_radius_m = 300


# -----------------------
# Helpers
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
    r = requests.get(url, params=params or {}, headers=headers or {}, allow_redirects=True, timeout=30)
    r.raise_for_status()
    data = r.content
    # verify via PIL (retry once)
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
# Google Places / Street View
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

    # Places Photos
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

    # Street View
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
# SerpAPI thumbnails (reference only, not exported)
# -----------------------
def serpapi_thumbnails(serp_key: str, query: str, limit: int = 6) -> List[Dict]:
    out: List[Dict] = []
    if not serp_key:
        return out
    try:
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
            img = None
            try:
                img = _download_image(thumb, headers={"User-Agent": "Mozilla/5.0"})
            except Exception:
                pass
            out.append({
                "title": item.get("title") or query,
                "source": "SerpAPI (reference)",
                "preview_bytes": img,  # may be None
                "usable": False,
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
# Export (webp)
# -----------------------
def render_to_webp(data_bytes: bytes, out_w=1200, out_h=675, quality=82) -> bytes:
    img = Image.open(io.BytesIO(data_bytes)).convert("RGB")
    img = crop_to_aspect(img, out_w, out_h).resize((out_w, out_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=quality, method=6)
    return buf.getvalue()

def export_selected(kw: str, cand: Dict, webp_quality: int = 82):
    if not cand.get("usable"):
        st.warning("This candidate is reference-only and cannot be exported.")
        return
    webp = render_to_webp(cand["preview_bytes"], quality=webp_quality)
    fname = f"{slugify(kw)}.webp"
    st.session_state.generated.setdefault(kw, [])
    st.session_state.generated[kw].append((fname, webp))
    st.download_button("⬇️ Download image", data=webp, file_name=fname, mime="image/webp")


# -----------------------
# AI Render (OpenAI Images)
# -----------------------
SITE_PROFILES = {
    "vailvacay.com":  "Photorealistic alpine resort & village scenes in the Colorado Rockies; gondolas; spruce & aspens; no text.",
    "bangkokvacay.com":"Photorealistic Bangkok city scenes; temples, night markets, skyline; warm light; no text.",
    "bostonvacay.com": "Photorealistic Boston; brick streets, brownstones, harbor; fall or winter appropriately; no text.",
    "ipetzo.com":      "Photorealistic pet lifestyle images; dogs/cats with owners indoors/outdoors; neutral decor; no logos; no text.",
    "1-800deals.com":  "Photorealistic retail/ecommerce visuals; shopping scenes, parcels, generic products; clean backgrounds; no brands; no text.",
}

def build_prompt(site: str, keyword: str, season_aware: bool) -> str:
    base = SITE_PROFILES.get(site, SITE_PROFILES["vailvacay.com"])
    style = "Balanced composition; natural light; editorial stock-photo feel. Landscape orientation. No words or typography."
    season = ""
    k = keyword.lower()
    if season_aware:
        if any(m in k for m in ["december","january","february","christmas","winter"]):
            season = " Winter scene with snow if outdoors; appropriate clothing."
        elif any(m in k for m in ["september","october","november","fall","autumn"]):
            season = " Autumn foliage if outdoors."
        elif any(m in k for m in ["may","april","spring"]):
            season = " Spring light and greenery if outdoors."
        elif any(m in k for m in ["june","july","august","summer"]):
            season = " Summer look if outdoors."
    return f"{base} Create an image for the topic: '{keyword}'. {style}{season}"

def lsi_variants(keyword: str, n: int, method: str) -> List[str]:
    if n <= 1 or method == "None":
        return [keyword]
    # very light heuristic expansions
    seeds = [
        "guide", "best", "tips", "near me", "with kids", "at night",
        "on a budget", "insider", "local", "itinerary", "photo spots"
    ]
    out = [keyword]
    i = 0
    while len(out) < n and i < len(seeds):
        out.append(f"{keyword} — {seeds[i]}")
        i += 1
    return out

def openai_generate_image_b64(prompt: str, size: str, api_key: str) -> bytes:
    # OpenAI Images: v1/images/generations with response_format=b64_json
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-image-1",
        "prompt": prompt,
        "size": size,
        "response_format": "b64_json"
    }
    r = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    b64 = r.json()["data"][0]["b64_json"]
    import base64
    return base64.b64decode(b64)


# -----------------------
# Sidebar
# -----------------------
st.sidebar.markdown("## Mode")
mode = st.sidebar.radio("", ["Real Photos", "AI Render"], index=0)

st.sidebar.markdown("## Keys")
g_key = st.sidebar.text_input("Google Maps/Places API key", type="password",
                              help="Required in Real Photos mode (Places API + Street View Static API enabled).")
serp_key = st.sidebar.text_input("SerpAPI key (optional)", type="password",
                                 help="Used only for reference thumbnails in Real Photos mode.")
openai_key = st.sidebar.text_input("OpenAI API key (for AI Render)", type="password",
                                   help="Required in AI Render mode.")

st.sidebar.markdown("## Output")
webp_q = st.sidebar.slider("WebP quality", 60, 95, 82)

if mode == "Real Photos":
    st.sidebar.markdown("## Sources to use")
    use_places = st.sidebar.checkbox("Google Places Photos", value=True)
    use_sv = st.sidebar.checkbox("Google Street View", value=True)
    use_serp = st.sidebar.checkbox("SerpAPI thumbnails (reference only)", value=False)

    st.sidebar.markdown("### Street View")
    sv_radius = st.sidebar.slider("Search radius (meters)", 50, 1000, st.session_state.sv_radius_m, step=50)
    st.session_state.sv_radius_m = sv_radius

    build_zip_btn = st.sidebar.button("Build ZIP of last exports")

else:
    st.sidebar.markdown("## AI settings")
    site = st.sidebar.selectbox("Site style", list(SITE_PROFILES.keys()), index=0)
    base_size = st.sidebar.selectbox("Render base size", ["1536x1024","1024x1536","1024x1024"], index=0)
    season_aware = st.sidebar.checkbox("Season-aware prompts", value=True)
    images_per_kw = st.sidebar.number_input("Images per keyword (LSI expansion)", min_value=1, max_value=10, value=1, step=1)
    lsi_method = st.sidebar.selectbox("LSI method", ["None","Heuristic"], index=1)

    build_zip_btn = st.sidebar.button("Build ZIP of last renders")


# -----------------------
# Main UI
# -----------------------
st.title("ImageForge v1.2 — Real Photos + AI Render")
st.caption("Pick a mode in the sidebar. Real Photos uses Google Places/Street View (plus optional SerpAPI reference). AI Render uses OpenAI Images with site-aware prompts.")

keywords_text = st.text_area("Paste keywords (one per line)", height=150,
                             placeholder="Blue Moose Pizza, Vail Colorado\nBest seafood restaurant in Boston")

colA, colB = st.columns([1,1])
go = colA.button("Generate candidates")
if colB.button("Clear"):
    st.session_state.candidates.clear()
    st.session_state.generated.clear()
    st.experimental_rerun()


# -----------------------
# Collect candidates
# -----------------------
def collect_real_photo_candidates(kw: str) -> List[Dict]:
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
    for i, c in enumerate(cands):
        c["label_idx"] = i
    return cands

def collect_ai_candidates(kw: str) -> List[Dict]:
    if not openai_key:
        st.warning("Enter your OpenAI API key in the sidebar.")
        return []
    prompts = lsi_variants(kw, images_per_kw, lsi_method)
    out: List[Dict] = []
    with st.spinner(f"Rendering with OpenAI for: {kw}"):
        for p in prompts:
            prompt = build_prompt(site, p, season_aware)
            try:
                png = openai_generate_image_b64(prompt, size=base_size, api_key=openai_key)
                out.append({
                    "title": p,
                    "source": "OpenAI Render",
                    "preview_bytes": png,
                    "usable": True,
                    "meta": {"kind":"openai","prompt": prompt}
                })
            except Exception as e:
                st.error(f"{p}: {e}")
    for i, c in enumerate(out):
        c["label_idx"] = i
    return out

if go:
    kws = [ln.strip() for ln in (keywords_text or "").splitlines() if ln.strip()]
    if not kws:
        st.warning("Please paste at least one keyword.")
    else:
        for kw in kws:
            if mode == "Real Photos":
                st.session_state.candidates[kw] = collect_real_photo_candidates(kw)
            else:
                st.session_state.candidates[kw] = collect_ai_candidates(kw)


# -----------------------
# Display candidates & Create Image
# -----------------------
def show_candidates_for_keyword(kw: str, candidates: List[Dict]):
    st.markdown(f"## {kw}")
    if not candidates:
        st.info("No candidates found.")
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

            lic = (c.get("meta") or {}).get("license_note")
            cred = (c.get("meta") or {}).get("credit")
            if lic: st.caption(lic)
            if cred: st.caption(f"Credit: {cred}")

            # radio to pick
            st.radio("Pick", options=[f"Use {i}", "Skip"],
                     index=0 if st.session_state[pick_key] == i else 1,
                     key=f"pick_radio_{kw}_{i}",
                     on_change=lambda k=pick_key, idx=i: st.session_state.__setitem__(k, idx))

            # per-card create image
            if st.button("Create Image", key=f"create_{kw}_{i}"):
                st.session_state[pick_key] = i
                export_selected(kw, c, webp_quality=webp_q)

    # Show last created for kw
    if kw in st.session_state.generated and st.session_state.generated[kw]:
        st.success("Created images for this keyword:")
        for fname, data in st.session_state.generated[kw]:
            st.download_button(f"⬇️ {fname}", data=data, file_name=fname, mime="image/webp")

for kw, cands in st.session_state.candidates.items():
    show_candidates_for_keyword(kw, cands)


# -----------------------
# ZIP export
# -----------------------
if build_zip_btn:
    if not st.session_state.generated:
        st.warning("You haven’t created any images yet.")
    else:
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
            for kw, items in st.session_state.generated.items():
                for fname, data in items:
                    zf.writestr(fname, data)
        mem.seek(0)
        st.download_button("⬇️ Download ZIP", data=mem,
                           file_name="imageforge_exports.zip",
                           mime="application/zip")
