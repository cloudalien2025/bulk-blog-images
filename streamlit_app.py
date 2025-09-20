# ImageForge v1.1 — Real-First + References + LSI
# Streamlit app for bulk blog images:
# - Real photos first: Google Places Photos -> Street View -> Openverse (CC)
# - Optional SerpAPI reference thumbnails (never exported to the ZIP)
# - Optional OpenAI fallback if absolutely no real photo is found
# - Heuristic LSI to produce multiple related images per keyword
# - Exports 1200x675 WebP (+ optional Pinterest 1000x1500) and metadata.csv

import io, os, re, zipfile, time, json, math, base64
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import requests
from PIL import Image
import streamlit as st
import pandas as pd

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="ImageForge v1.1 — Real-First + References + LSI", layout="wide")

APP_TITLE = "ImageForge v1.1 — Real-First + References + LSI"
TARGET_W, TARGET_H = 1200, 675
PINTEREST_W, PINTEREST_H = 1000, 1500
DEFAULT_QUALITY = 82
ALLOWED_RENDER_SIZES = ["1536x1024", "1024x1536", "1024x1024"]  # for AI fallback

# Site style hints (used only for AI fallback prompts)
SITE_PROFILES = {
    "vailvacay.com":  "Colorado Rockies ski resort & alpine village scenes; gondolas; evergreen forests; creeks; cozy lodges.",
    "bangkokvacay.com":"Bangkok street & skyline; temples, markets, canals, BTS/MRT; rooftop views; tropical light.",
    "bostonvacay.com": "New England city: brick townhouses, harbor views, cobblestones, fall colors, historic districts.",
    "ipetzo.com":      "Pet lifestyle: dogs/cats with owners indoors/outdoors; parks, trails; tasteful, no brands/logos.",
    "1-800deals.com":  "Retail scenes: parcels, simple product flats, shopping vibes; neutral backgrounds; no brands/logos."
}
DEFAULT_SITE = "vailvacay.com"

# ---------------------------
# Utilities
# ---------------------------

def slugify(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[’'`]", "", t)
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t or "image"

def crop_to_aspect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    target_ratio = target_w / target_h
    w, h = img.size
    cur_ratio = w / h if h else 1
    if cur_ratio > target_ratio:
        new_w = int(h * target_ratio); left = (w - new_w) // 2
        box = (left, 0, left + new_w, h)
    else:
        new_h = int(w / target_ratio); top = (h - new_h) // 2
        box = (0, top, w, top + new_h)
    return img.crop(box)

def to_webp_bytes(img_bytes: bytes, target_w: int, target_h: int, quality: int) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = crop_to_aspect(img, target_w, target_h).resize((target_w, target_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=quality, method=6)
    return buf.getvalue()

def get_image_bytes(url: str, timeout: int = 25) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.ok and r.content:
            return r.content
    except Exception:
        pass
    return None

@dataclass
class Candidate:
    preview_bytes: bytes
    full_bytes: bytes
    source: str
    title: str
    license: str
    credit: str
    credit_url: Optional[str]
    is_reference_only: bool = False   # never added to ZIP when True

# ---------------------------
# Sources: Google Places, Street View, Openverse, SerpAPI
# ---------------------------

def google_places_candidates(query: str, key: str, max_photos: int = 4) -> List[Candidate]:
    """Get venue photos from Google Places Text Search."""
    out = []
    try:
        ts_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        r = requests.get(ts_url, params={"query": query, "key": key}, timeout=20)
        data = r.json()
        if not data.get("results"): 
            return out

        place = data["results"][0]
        photos = place.get("photos", [])[:max_photos]
        name = place.get("name", query)
        place_url = f"https://www.google.com/maps/place/?q=place_id:{place.get('place_id')}"
        for ph in photos:
            ref = ph.get("photo_reference")
            if not ref: 
                continue
            # photo endpoint redirects to the actual image; requests follows redirects by default
            purl = "https://maps.googleapis.com/maps/api/place/photo"
            img = get_image_bytes(purl + f"?maxwidth=1600&photo_reference={ref}&key={key}")
            thumb = img
            if img:
                credit_html = "; ".join(ph.get("html_attributions", [])) or "Google Places Photo"
                out.append(Candidate(
                    preview_bytes=thumb,
                    full_bytes=img,
                    source="Google Places Photo",
                    title=name,
                    license="Refer to Google Places Photo terms",
                    credit=credit_html,
                    credit_url=place_url,
                ))
    except Exception:
        pass
    return out

def google_streetview_candidate(query: str, key: str, lat: float = None, lng: float = None) -> List[Candidate]:
    """If Places search succeeds, use its location; else geocode by query (Places again)."""
    out = []
    try:
        if lat is None or lng is None:
            ts_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            r = requests.get(ts_url, params={"query": query, "key": key}, timeout=20)
            data = r.json()
            if not data.get("results"):
                return out
            loc = data["results"][0]["geometry"]["location"]
            lat, lng = loc["lat"], loc["lng"]

        base = "https://maps.googleapis.com/maps/api/streetview"
        # 640x640 is safe on the free tier
        url = f"{base}?size=640x640&location={lat},{lng}&key={key}"
        img = get_image_bytes(url)
        if img:
            out.append(Candidate(
                preview_bytes=img, full_bytes=img,
                source="Google Street View",
                title=query,
                license="Refer to Google Street View terms",
                credit="Google Street View",
                credit_url=None
            ))
    except Exception:
        pass
    return out

def openverse_cc_candidates(query: str, max_results: int = 6) -> List[Candidate]:
    """Public CC images (commercial-friendly) via Openverse."""
    out = []
    try:
        url = "https://api.openverse.engineering/v1/images/"
        params = {"q": query, "license_type": "commercial", "page_size": max_results}
        r = requests.get(url, params=params, timeout=25)
        if not r.ok:
            return out
        for item in r.json().get("results", []):
            img_url = item.get("url") or item.get("thumbnail")
            if not img_url:
                continue
            img = get_image_bytes(img_url)
            if not img:
                continue
            title = item.get("title") or query
            creator = item.get("creator") or "Unknown"
            lic = item.get("license") or "CC"
            credit_url = item.get("foreign_landing_url") or item.get("url")
            out.append(Candidate(
                preview_bytes=img, full_bytes=img,
                source="Openverse (CC)",
                title=title,
                license=lic,
                credit=f"{creator} • CC {lic}",
                credit_url=credit_url
            ))
    except Exception:
        pass
    return out

def serpapi_reference_images(query: str, key: str, n: int = 6) -> List[Candidate]:
    """Reference-only thumbnails from Google Images via SerpAPI."""
    out = []
    try:
        url = "https://serpapi.com/search.json"
        params = {"engine": "google", "q": query, "tbm": "isch", "num": n, "api_key": key}
        r = requests.get(url, params=params, timeout=25)
        if not r.ok:
            return out
        for img in r.json().get("images_results", [])[:n]:
            img_url = img.get("original") or img.get("thumbnail")
            if not img_url:
                continue
            data = get_image_bytes(img_url)
            if not data:
                continue
            title = img.get("title") or query
            credit_url = img.get("link")
            out.append(Candidate(
                preview_bytes=data, full_bytes=data,
                source="SerpAPI (Google Images, reference)",
                title=title,
                license="Unknown / reference-only",
                credit="Google Images via SerpAPI — reference only",
                credit_url=credit_url,
                is_reference_only=True
            ))
    except Exception:
        pass
    return out

# ---------------------------
# AI Fallback (only if needed)
# ---------------------------

def build_ai_prompt(site: str, keyword: str) -> str:
    base = SITE_PROFILES.get(site, SITE_PROFILES[DEFAULT_SITE])
    return (
        f"Photorealistic travel blog image. {base} "
        f"Create a landscape scene that makes sense for the topic: '{keyword}'. "
        f"Balanced composition, natural light, editorial stock-photo feel, no text or logos."
    )

def openai_generate_image_bytes(prompt: str, size: str, api_key: str) -> Optional[bytes]:
    """Robust call to OpenAI Images with graceful fallbacks for response format."""
    try:
        url = "https://api.openai.com/v1/images/generations"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        payload = {"model":"gpt-image-1","prompt":prompt,"size":size}  # keep default response (URL)
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            d = r.json().get("data", [{}])[0]
            if "url" in d:
                img = get_image_bytes(d["url"], timeout=60)
                if img: return img
            if "b64_json" in d:
                return base64.b64decode(d["b64_json"])

        # If server complains about params, try explicit b64 mode
        payload = {"model":"gpt-image-1","prompt":prompt,"size":size,"response_format":"b64_json"}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            d = r.json().get("data", [{}])[0]
            if "b64_json" in d:
                return base64.b64decode(d["b64_json"])
    except Exception:
        pass
    return None

# ---------------------------
# LSI (heuristic) expansion
# ---------------------------

def lsi_expand(keyword: str, count: int) -> List[str]:
    """Very light heuristic LSI variants; keeps the topic but nudges wording."""
    if count <= 1:
        return [keyword]
    base = keyword.strip()
    seeds = [
        base,
        base.replace("things to do", "top activities"),
        base.replace("in ", "near ").replace("at ", "near "),
        re.sub(r"\bhow to\b", "guide to", base, flags=re.I),
        re.sub(r"\bwhat to\b", "where to", base, flags=re.I),
        base + " tips",
        base + " ideas",
        base.replace("best ", "").replace(" top ", " best "),
        "scenic " + base if not base.lower().startswith("scenic") else base + " guide",
        base + " for families",
        base + " for couples",
        base + " map",
        base + " itinerary",
    ]
    out = []
    for s in seeds:
        if s not in out:
            out.append(s)
        if len(out) >= count:
            break
    return out[:count]

# ---------------------------
# UI — Sidebar
# ---------------------------

st.title(APP_TITLE)
st.caption("Real photos first (Places/Street View/Openverse). Optional SerpAPI reference thumbnails. Optional OpenAI fallback only when no real photo exists.")

with st.sidebar:
    st.header("Keys")
    google_key = st.text_input("Google Maps/Places API key (required for real photos)", type="password")
    openai_key = st.text_input("OpenAI API key (optional, AI fallback)", type="password")
    serpapi_key = st.text_input("SerpAPI key (optional, reference thumbnails)", type="password")

    st.header("Output")
    site = st.selectbox("Site style (AI fallback only)", list(SITE_PROFILES.keys()),
                        index=list(SITE_PROFILES.keys()).index(DEFAULT_SITE))
    base_size = st.selectbox("Render base size (source fetch/crop target)", ALLOWED_RENDER_SIZES, index=0)
    quality = st.slider("WebP quality", 60, 95, DEFAULT_QUALITY)
    also_pinterest = st.checkbox("Also make a Pinterest image (1000×1500)", value=False)

    st.header("Sources to use")
    use_places = st.checkbox("Google Places Photos", value=True)
    use_street = st.checkbox("Google Street View", value=True)
    use_openverse = st.checkbox("Openverse (CC)", value=True)
    use_serpapi_refs = st.checkbox("SerpAPI thumbnails (reference only)", value=True if serpapi_key else False)

    st.header("Picker & LSI")
    manual_pick = st.selectbox("Thumbnail picking", ["Manual pick thumbnails", "Auto-best"], index=0)
    imgs_per_kw = st.number_input("Images per keyword (LSI expansion)", min_value=1, max_value=20, value=1)
    lsi_method = st.selectbox("LSI method", ["Heuristic"], index=0)

    st.header("Fallbacks")
    season_aware = st.checkbox("Season-aware prompts (AI fallback)", value=True)
    allow_ai_fallback = st.checkbox("Allow AI fallback if no real photo found", value=False)

# ---------------------------
# Main input
# ---------------------------
st.subheader("Paste keywords (one per line)")
kw_text = st.text_area("", height=180, placeholder="Tavern on the Square, Vail Colorado\nBest seafood restaurant in Boston\nThings to do in Vail in October")
colA, colB = st.columns([1,1])
run_btn = colA.button("Generate")
clr_btn = colB.button("Clear")
if clr_btn:
    st.session_state.clear()
    st.experimental_rerun()

# ---------------------------
# Processing
# ---------------------------
def collect_candidates_for_query(q: str) -> List[Candidate]:
    candidates: List[Candidate] = []

    # 1) Google Places Photos
    if use_places and google_key:
        candidates += google_places_candidates(q, google_key, max_photos=4)

    # 2) Street View
    if use_street and google_key:
        candidates += google_streetview_candidate(q, google_key)

    # 3) Openverse CC
    if use_openverse:
        candidates += openverse_cc_candidates(q, max_results=6)

    # 4) SerpAPI references (reference-only)
    if use_serpapi_refs and serpapi_key:
        candidates += serpapi_reference_images(q, serpapi_key, n=6)

    return candidates

def choose_best(cands: List[Candidate]) -> Optional[int]:
    """Very light scorer: prioritize non-reference Google Places/Openverse, avoid Street View last."""
    if not cands:
        return None
    best_idx, best_score = 0, -1
    for i, c in enumerate(cands):
        score = 0
        if c.is_reference_only: score -= 5
        if c.source.startswith("Google Places"): score += 5
        if c.source.startswith("Openverse"): score += 3
        if c.source.startswith("Google Street View"): score += 1
        # size heuristic
        try:
            im = Image.open(io.BytesIO(c.preview_bytes))
            score += min(im.size[0], im.size[1]) / 500.0
        except Exception:
            pass
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx

def ai_image_for(q: str) -> Optional[bytes]:
    if not openai_key or not allow_ai_fallback:
        return None
    # Season-aware nudge (simple): month words hint winter vs summer
    seasonal = ""
    if season_aware:
        qlow = q.lower()
        winter_words = ["december", "january", "february", "christmas", "snow", "ski", "november"]
        summer_words = ["june", "july", "august", "summer"]
        fall_words = ["september", "october", "fall", "autumn"]
        spring_words = ["april", "may", "spring", "blossom"]
        if any(w in qlow for w in winter_words):
            seasonal = " Winter setting: snow present if realistic."
        elif any(w in qlow for w in summer_words):
            seasonal = " Summer setting."
        elif any(w in qlow for w in fall_words):
            seasonal = " Autumn foliage likely."
        elif any(w in qlow for w in spring_words):
            seasonal = " Spring shoulder season."
    prompt = build_ai_prompt(site, q) + seasonal
    img_bytes = openai_generate_image_bytes(prompt, size=base_size, api_key=openai_key)
    return img_bytes

def save_rows_to_zip(zipf, rows, pinterest: bool):
    for row in rows:
        # Skip reference-only rows for the ZIP
        if row["reference_only"]:
            continue
        webp_bytes = to_webp_bytes(row["raw_bytes"], TARGET_W, TARGET_H, quality)
        zipf.writestr(row["filename"], webp_bytes)
        if pinterest:
            pin_bytes = to_webp_bytes(row["raw_bytes"], PINTEREST_W, PINTEREST_H, quality)
            base = row["filename"].rsplit(".", 1)[0]
            zipf.writestr(f"{base}_pin.webp", pin_bytes)

# ---------------------------
# Run
# ---------------------------
if run_btn:
    keywords = [ln.strip() for ln in kw_text.splitlines() if ln.strip()]
    if not keywords:
        st.warning("Please paste at least one keyword.")
        st.stop()

    if (use_places or use_street) and not google_key:
        st.warning("Google Maps/Places API key is required for real photos (Places/Street View).")
        st.stop()

    all_rows = []
    zip_buf = io.BytesIO()
    zipf = zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED)
    progress = st.progress(0)
    status = st.empty()

    # metadata rows
    meta_rows = []

    # expand with LSI if needed
    expanded_keywords = []
    for kw in keywords:
        expanded_keywords.extend(lsi_expand(kw, imgs_per_kw))

    for idx, kw in enumerate(expanded_keywords, start=1):
        status.info(f"Working {idx}/{len(expanded_keywords)}: {kw}")
        cands = collect_candidates_for_query(kw)

        # Show candidates block
        with st.expander(f"📷 Thumbnails — choose one or leave 'Auto-best': {kw}", expanded=False):
            picked_idx = None
            if manual_pick == "Manual pick thumbnails" and cands:
                # show gallery with radios
                cols = st.columns(3)
                radios = []
                for i, c in enumerate(cands):
                    with cols[i % 3]:
                        st.image(c.preview_bytes, use_container_width=True, caption=f"{c.source} — {c.title}")
                        st.caption(f"License: {c.license}\n\nCredit: {c.credit}")
                        picked_i = st.radio("Pick", [f"Use {i}", "Skip"], index=1, key=f"pick_{kw}_{i}")
                        radios.append((i, picked_i))
                for i, val in radios:
                    if val == f"Use {i}":
                        picked_idx = i
                        break

            # If manual pick left none, choose best automatically
            if picked_idx is None:
                picked_idx = choose_best(cands)

            if picked_idx is not None:
                c = cands[picked_idx]
                ok_img = True
                row = {
                    "keyword": kw,
                    "source": c.source,
                    "license": c.license,
                    "credit": c.credit,
                    "credit_url": c.credit_url,
                    "reference_only": c.is_reference_only,
                    "raw_bytes": c.full_bytes,
                    "filename": f"{slugify(kw)}.webp"
                }
                # Only export if not reference-only
                all_rows.append(row)
                meta_rows.append({
                    "keyword": kw,
                    "chosen_source": c.source,
                    "license": c.license,
                    "credit": c.credit,
                    "credit_url": c.credit_url,
                    "reference_only": c.is_reference_only
                })
            else:
                # No real candidate chosen; try AI fallback if allowed
                ai_img = ai_image_for(kw)
                if ai_img:
                    all_rows.append({
                        "keyword": kw,
                        "source": f"OpenAI ({base_size})",
                        "license": "OpenAI output",
                        "credit": "Generated via OpenAI — no text/logos",
                        "credit_url": None,
                        "reference_only": False,
                        "raw_bytes": ai_img,
                        "filename": f"{slugify(kw)}.webp"
                    })
                    meta_rows.append({
                        "keyword": kw,
                        "chosen_source": f"OpenAI ({base_size})",
                        "license": "OpenAI output",
                        "credit": "Generated via OpenAI — no text/logos",
                        "credit_url": None,
                        "reference_only": False
                    })
                else:
                    st.warning(f"No usable photo for: **{kw}** (and AI fallback disabled/unavailable).")

        # Each step progress
        progress.progress(idx / len(expanded_keywords))

    # Write ZIP
    save_rows_to_zip(zipf, all_rows, also_pinterest)
    # metadata CSV
    if meta_rows:
        df = pd.DataFrame(meta_rows)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        zipf.writestr("metadata.csv", csv_bytes)
    zipf.close()
    zip_buf.seek(0)

    st.success("Done! Download your bundle below.")
    st.download_button("⬇️ Download ZIP", data=zip_buf, file_name="imageforge_bundle.zip", mime="application/zip")

    # Previews
    st.markdown("### Previews & individual downloads")
    cols = st.columns(3)
    for i, row in enumerate(all_rows):
        if row["reference_only"]:
            continue
        with cols[i % 3]:
            show_bytes = to_webp_bytes(row["raw_bytes"], TARGET_W, TARGET_H, quality)
            st.image(show_bytes, use_container_width=True, caption=row["filename"])
            st.download_button("Download", data=show_bytes, file_name=row["filename"], mime="image/webp")

# ---------------------------
# Footer notes
# ---------------------------
st.markdown(
"""
**Notes & Tips**
- *Google Places Photos* and *Street View* have their own usage terms; include the provided credit metadata where appropriate.
- *SerpAPI thumbnails* are for **reference only** and are never exported to your ZIP.
- *Openverse (CC)* results include license info and landing links in `metadata.csv`.
- Use *AI fallback* only when necessary; it crops from the closest allowed size to 1200×675.
"""
)
