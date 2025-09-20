# ImageForge v1.2 — Real-First + References + LSI (manual pick = 2-step, one choice per keyword)
# - Step 1: Generate -> show candidates (Places/Street View/Openverse/SerpAPI refs)
# - Step 2: Pick one per keyword (single select) -> Build images from my selections
# - Strict image-byte validation to avoid crashes
# - 1200x675 WebP (+ optional 1000x1500 Pinterest), metadata.csv
# - LSI expansion (heuristic), optional OpenAI fallback

import io, re, zipfile, base64
from typing import List, Optional
from dataclasses import dataclass

import requests
from PIL import Image
import streamlit as st
import pandas as pd

st.set_page_config(page_title="ImageForge v1.2 — Real-First + References + LSI", layout="wide")

APP_TITLE = "ImageForge v1.2 — Real-First + References + LSI"
TARGET_W, TARGET_H = 1200, 675
PINTEREST_W, PINTEREST_H = 1000, 1500
DEFAULT_QUALITY = 82
ALLOWED_RENDER_SIZES = ["1536x1024", "1024x1536", "1024x1024"]

SITE_PROFILES = {
    "vailvacay.com":  "Colorado Rockies ski resort & alpine village scenes; gondolas; evergreen forests; creeks; cozy lodges.",
    "bangkokvacay.com":"Bangkok street & skyline; temples, markets, canals, BTS/MRT; rooftop views; tropical light.",
    "bostonvacay.com": "New England city: brick townhouses, harbor views, cobblestones, fall colors, historic districts.",
    "ipetzo.com":      "Pet lifestyle: dogs/cats with owners indoors/outdoors; parks, trails; tasteful, no brands/logos.",
    "1-800deals.com":  "Retail scenes: parcels, simple product flats, shopping vibes; neutral backgrounds; no brands/logos."
}
DEFAULT_SITE = "vailvacay.com"

# ---------------- Utils ----------------

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

def is_image_bytes(b: Optional[bytes]) -> bool:
    if not b:
        return False
    try:
        im = Image.open(io.BytesIO(b))
        im.verify()
        return True
    except Exception:
        return False

def to_webp_bytes(img_bytes: bytes, target_w: int, target_h: int, quality: int) -> bytes:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = crop_to_aspect(img, target_w, target_h).resize((target_w, target_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=quality, method=6)
    return buf.getvalue()

def get_image_bytes(url: str, timeout: int = 25) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=timeout, allow_redirects=True)
        if not r.ok or not r.content:
            return None
        ct = r.headers.get("content-type", "")
        data = r.content
        if ct.startswith("image/") or is_image_bytes(data):
            return data
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
    is_reference_only: bool = False

def valid_candidate(c: Optional[Candidate]) -> bool:
    return (
        c is not None
        and is_image_bytes(c.preview_bytes)
        and is_image_bytes(c.full_bytes)
    )

# ---------------- Sources ----------------

def google_places_candidates(query: str, key: str, max_photos: int = 4) -> List[Candidate]:
    out: List[Candidate] = []
    try:
        ts_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        r = requests.get(ts_url, params={"query": query, "key": key}, timeout=20)
        data = r.json()
        results = data.get("results", [])
        if not results:
            return out
        place = results[0]
        name = place.get("name", query)
        place_url = f"https://www.google.com/maps/place/?q=place_id:{place.get('place_id')}"
        photos = place.get("photos", [])[:max_photos]
        for ph in photos:
            ref = ph.get("photo_reference")
            if not ref:
                continue
            purl = "https://maps.googleapis.com/maps/api/place/photo"
            img = get_image_bytes(f"{purl}?maxwidth=1600&photo_reference={ref}&key={key}")
            if not is_image_bytes(img):
                continue
            credit_html = "; ".join(ph.get("html_attributions", [])) or "Google Places Photo"
            cand = Candidate(
                preview_bytes=img, full_bytes=img,
                source="Google Places Photo",
                title=name,
                license="Refer to Google Places Photo terms",
                credit=credit_html,
                credit_url=place_url,
            )
            if valid_candidate(cand):
                out.append(cand)
    except Exception:
        pass
    return out

def google_streetview_candidate(query: str, key: str) -> List[Candidate]:
    out: List[Candidate] = []
    try:
        ts_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        r = requests.get(ts_url, params={"query": query, "key": key}, timeout=20)
        data = r.json()
        results = data.get("results", [])
        if not results:
            return out
        loc = results[0]["geometry"]["location"]
        lat, lng = loc["lat"], loc["lng"]
        base = "https://maps.googleapis.com/maps/api/streetview"
        url = f"{base}?size=640x640&location={lat},{lng}&key={key}"
        img = get_image_bytes(url)
        if not is_image_bytes(img):
            return out
        cand = Candidate(
            preview_bytes=img, full_bytes=img,
            source="Google Street View",
            title=query,
            license="Refer to Google Street View terms",
            credit="Google Street View",
            credit_url=None
        )
        if valid_candidate(cand):
            out.append(cand)
    except Exception:
        pass
    return out

def openverse_cc_candidates(query: str, max_results: int = 6) -> List[Candidate]:
    out: List[Candidate] = []
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
            if not is_image_bytes(img):
                continue
            title = item.get("title") or query
            creator = item.get("creator") or "Unknown"
            lic = item.get("license") or "CC"
            credit_url = item.get("foreign_landing_url") or item.get("url")
            cand = Candidate(
                preview_bytes=img, full_bytes=img,
                source="Openverse (CC)",
                title=title,
                license=lic,
                credit=f"{creator} • CC {lic}",
                credit_url=credit_url
            )
            if valid_candidate(cand):
                out.append(cand)
    except Exception:
        pass
    return out

def serpapi_reference_images(query: str, key: str, n: int = 6) -> List[Candidate]:
    out: List[Candidate] = []
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
            if not is_image_bytes(data):
                continue
            title = img.get("title") or query
            credit_url = img.get("link")
            cand = Candidate(
                preview_bytes=data, full_bytes=data,
                source="SerpAPI (Google Images, reference)",
                title=title,
                license="Unknown / reference-only",
                credit="Google Images via SerpAPI — reference only",
                credit_url=credit_url,
                is_reference_only=True
            )
            if valid_candidate(cand):
                out.append(cand)
    except Exception:
        pass
    return out

# ---------------- AI fallback ----------------

def build_ai_prompt(site: str, keyword: str) -> str:
    base = SITE_PROFILES.get(site, SITE_PROFILES[DEFAULT_SITE])
    return (
        f"Photorealistic travel blog image. {base} "
        f"Create a landscape scene that makes sense for the topic: '{keyword}'. "
        f"Balanced composition, natural light, editorial stock-photo feel, no text or logos."
    )

def openai_generate_image_bytes(prompt: str, size: str, api_key: str) -> Optional[bytes]:
    try:
        url = "https://api.openai.com/v1/images/generations"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        # Try URL first
        payload = {"model": "gpt-image-1", "prompt": prompt, "size": size}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            d = r.json().get("data", [{}])[0]
            if "url" in d:
                img = get_image_bytes(d["url"], timeout=60)
                if is_image_bytes(img):
                    return img
            if "b64_json" in d:
                data = base64.b64decode(d["b64_json"])
                if is_image_bytes(data):
                    return data

        # Fallback to explicit b64
        payload = {"model":"gpt-image-1","prompt":prompt,"size":size,"response_format":"b64_json"}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            d = r.json().get("data", [{}])[0]
            if "b64_json" in d:
                data = base64.b64decode(d["b64_json"])
                if is_image_bytes(data):
                    return data
    except Exception:
        pass
    return None

# ---------------- LSI ----------------

def lsi_expand(keyword: str, count: int) -> List[str]:
    if count <= 1:
        return [keyword]
    base = keyword.strip()
    seeds = [
        base,
        base.replace("things to do", "top activities"),
        base.replace(" in ", " near ").replace(" at ", " near "),
        re.sub(r"\bhow to\b", "guide to", base, flags=re.I),
        re.sub(r"\bwhat to\b", "where to", base, flags=re.I),
        base + " tips",
        base + " ideas",
        base.replace("best ", "").replace(" top ", " best "),
        ("scenic " + base) if not base.lower().startswith("scenic") else (base + " guide"),
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

# ---------------- UI ----------------

st.title(APP_TITLE)
st.caption("Two-step manual pick: Generate → choose one per keyword → Build images. Real photos first, AI only if needed.")

with st.sidebar:
    st.header("Keys")
    google_key = st.text_input("Google Maps/Places API key (required for real photos)", type="password")
    openai_key = st.text_input("OpenAI API key (optional, AI fallback)", type="password")
    serpapi_key = st.text_input("SerpAPI key (optional, reference thumbnails)", type="password")

    st.header("Output")
    site = st.selectbox("Site style (AI fallback only)", list(SITE_PROFILES.keys()),
                        index=list(SITE_PROFILES.keys()).index(DEFAULT_SITE))
    base_size = st.selectbox("Render base size (AI fallback)", ALLOWED_RENDER_SIZES, index=0)
    quality = st.slider("WebP quality", 60, 95, DEFAULT_QUALITY)
    also_pinterest = st.checkbox("Also make a Pinterest image (1000×1500)", value=False)

    st.header("Sources to use")
    use_places = st.checkbox("Google Places Photos", value=True)
    use_street = st.checkbox("Google Street View", value=True)
    use_openverse = st.checkbox("Openverse (CC)", value=True)
    use_serpapi_refs = st.checkbox("SerpAPI thumbnails (reference only)", value=True if serpapi_key else False)

    st.header("Picker & LSI")
    manual_pick = st.selectbox("Thumbnail picking", ["Manual pick thumbnails (2-step)", "Auto-best"], index=0)
    imgs_per_kw = st.number_input("Images per keyword (LSI expansion)", min_value=1, max_value=20, value=1)
    lsi_method = st.selectbox("LSI method", ["Heuristic"], index=0)

    st.header("Fallbacks")
    season_aware = st.checkbox("Season-aware prompts (AI fallback)", value=True)
    allow_ai_fallback = st.checkbox("Allow AI fallback if no real photo found", value=False)

st.subheader("Paste keywords (one per line)")
kw_text = st.text_area("", height=180, placeholder="Blue Moose Pizza in Vail Colorado\nTavern on the Square, Vail Colorado\nBest seafood restaurant in Boston")

colA, colB = st.columns([1,1])
generate_btn = colA.button("Generate")
clear_btn = colB.button("Clear")
if clear_btn:
    st.session_state.clear()
    st.experimental_rerun()

# ---------------- Collect ----------------

def collect_candidates_for_query(q: str) -> List[Candidate]:
    cands: List[Candidate] = []
    if use_places and google_key:
        cands += google_places_candidates(q, google_key, max_photos=4)
    if use_street and google_key:
        cands += google_streetview_candidate(q, google_key)
    if use_openverse:
        cands += openverse_cc_candidates(q, max_results=6)
    if use_serpapi_refs and serpapi_key:
        cands += serpapi_reference_images(q, serpapi_key, n=6)
    cands = [c for c in cands if valid_candidate(c)]
    return cands

def choose_best(cands: List[Candidate]) -> Optional[int]:
    if not cands:
        return None
    best_idx, best_score = 0, -1
    for i, c in enumerate(cands):
        score = 0
        if c.is_reference_only: score -= 5
        if c.source.startswith("Google Places"): score += 5
        if c.source.startswith("Openverse"): score += 3
        if c.source.startswith("Google Street View"): score += 1
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
    seasonal = ""
    if season_aware:
        qlow = q.lower()
        winter = ["december","january","february","christmas","snow","ski","november"]
        summer = ["june","july","august","summer"]
        fall = ["september","october","fall","autumn"]
        spring = ["april","may","spring","blossom"]
        if any(w in qlow for w in winter): seasonal = " Winter setting: snow present if realistic."
        elif any(w in qlow for w in summer): seasonal = " Summer setting."
        elif any(w in qlow for w in fall): seasonal = " Autumn foliage likely."
        elif any(w in qlow for w in spring): seasonal = " Spring shoulder season."
    prompt = build_ai_prompt(site, q) + seasonal
    img_bytes = openai_generate_image_bytes(prompt, size=base_size, api_key=openai_key)
    return img_bytes if is_image_bytes(img_bytes) else None

def build_zip(rows, pinterest: bool, quality: int) -> bytes:
    buf = io.BytesIO()
    z = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED)
    for r in rows:
        if r["reference_only"]:
            continue
        webp = to_webp_bytes(r["raw_bytes"], TARGET_W, TARGET_H, quality)
        z.writestr(r["filename"], webp)
        if pinterest:
            pin = to_webp_bytes(r["raw_bytes"], PINTEREST_W, PINTEREST_H, quality)
            base = r["filename"].rsplit(".", 1)[0]
            z.writestr(f"{base}_pin.webp", pin)
    if rows:
        meta = pd.DataFrame([{
            "keyword": r["keyword"],
            "chosen_source": r["source"],
            "license": r["license"],
            "credit": r["credit"],
            "credit_url": r["credit_url"],
            "reference_only": r["reference_only"],
        } for r in rows if not r["reference_only"]])
        z.writestr("metadata.csv", meta.to_csv(index=False).encode("utf-8"))
    z.close(); buf.seek(0)
    return buf.getvalue()

# ---------------- Run flow ----------------

if "phase" not in st.session_state:
    st.session_state.phase = "idle"   # idle | picking | building

if generate_btn:
    kws = [ln.strip() for ln in kw_text.splitlines() if ln.strip()]
    if not kws:
        st.warning("Please paste at least one keyword.")
        st.stop()
    if (use_places or use_street) and not google_key:
        st.warning("Google Maps/Places API key is required for real photos.")
        st.stop()
    # expand LSI now
    expanded = []
    for kw in kws:
        expanded.extend(lsi_expand(kw, imgs_per_kw))
    st.session_state.keywords = expanded
    st.session_state.phase = "picking"
    st.experimental_rerun()

if st.session_state.phase == "picking":
    st.info("Step 1/2 — Pick one thumbnail per keyword (or leave 'Auto-best'), then click **Build images from my selections**.")
    build_btn = st.button("Build images from my selections")

    for kw in st.session_state.get("keywords", []):
        cands = collect_candidates_for_query(kw)
        with st.expander(f"📷 Thumbnails — {kw}", expanded=False):
            if not cands:
                st.caption("No real-photo candidates. If allowed, AI fallback will be used at build time.")
            else:
                cols = st.columns(3)
                for i, c in enumerate(cands):
                    with cols[i % 3]:
                        try:
                            st.image(c.preview_bytes, use_container_width=True,
                                     caption=f"[{i}] {c.source} — {c.title}")
                        except Exception:
                            st.caption(f"[{i}] (unrenderable bytes, skipped)")
                        st.caption(f"License: {c.license}")
                        st.caption(f"Credit: {c.credit}")
            # single select per keyword
            options = ["Auto-best"] + [f"Use {i}" for i in range(len(cands))]
            st.selectbox("Your choice", options, key=f"choice_{slugify(kw)}")

    if build_btn:
        st.session_state.phase = "building"
        st.experimental_rerun()

if st.session_state.phase == "building":
    all_rows = []
    progress = st.progress(0)
    for idx, kw in enumerate(st.session_state.get("keywords", []), start=1):
        cands = collect_candidates_for_query(kw)
        choice = st.session_state.get(f"choice_{slugify(kw)}", "Auto-best")
        picked_idx = None
        if choice.startswith("Use "):
            try:
                picked_idx = int(choice.split("Use ")[1])
            except Exception:
                picked_idx = None
        if picked_idx is None:
            picked_idx = choose_best(cands)

        if picked_idx is not None and 0 <= picked_idx < len(cands):
            c = cands[picked_idx]
            all_rows.append({
                "keyword": kw,
                "source": c.source,
                "license": c.license,
                "credit": c.credit,
                "credit_url": c.credit_url,
                "reference_only": c.is_reference_only,
                "raw_bytes": c.full_bytes,
                "filename": f"{slugify(kw)}.webp"
            })
        else:
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
            else:
                st.warning(f"No usable photo for: **{kw}** (and AI fallback disabled/unavailable).")
        progress.progress(idx / max(1, len(st.session_state.get("keywords", []))))

    if all_rows:
        zip_bytes = build_zip(all_rows, also_pinterest, quality)
        st.success("Done! Download your bundle below.")
        st.download_button("⬇️ Download ZIP", data=zip_bytes,
                           file_name="imageforge_bundle.zip", mime="application/zip")

        st.markdown("### Previews & individual downloads")
        cols = st.columns(3)
        shown = 0
        for r in all_rows:
            if r["reference_only"]:
                continue
            try:
                show_bytes = to_webp_bytes(r["raw_bytes"], TARGET_W, TARGET_H, quality)
            except Exception:
                continue
            with cols[shown % 3]:
                st.image(show_bytes, use_container_width=True, caption=r["filename"])
                st.download_button("Download", data=show_bytes,
                                   file_name=r["filename"], mime="image/webp")
            shown += 1

    # reset to idle so the next run starts fresh
    st.session_state.phase = "idle"

st.markdown("""
**Notes**
- Manual pick is now a 2-step flow to ensure your selection is honored.
- SerpAPI thumbnails are reference-only (never exported).
- Openverse (CC) entries include license/credit in metadata.csv.
- All images are byte-validated before display/export to avoid crashes.
""")
