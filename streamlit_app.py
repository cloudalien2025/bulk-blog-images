# ImageForge v0.9.5 (Inspiration Mode)
# CC-first, safe non-CC "reference only" thumbnails, and an Inspiration mode
# that guides an original AI render from a non-CC reference without copying it.
# Requirements: streamlit, requests, pillow

import os, io, re, json, time, base64, zipfile
from typing import List, Dict, Optional, Tuple
import requests
from PIL import Image
import streamlit as st

APP_TITLE = "ImageForge – CC + Inspiration"

DEFAULT_OUTPUT_W, DEFAULT_OUTPUT_H = 1200, 675
DEFAULT_WEBP_QUALITY = 82
OPENAI_IMG_SIZES = ["1536x1024", "1024x1536", "1024x1024"]  # we'll crop to target

SITE_PROFILES = {
    "vailvacay.com":  "Photorealistic alpine resort & village scenes in the Colorado Rockies; gondolas, riverwalks, cozy lodges; no text.",
    "bangkokvacay.com":"Photorealistic Bangkok city life; street food, temples, river views, neon night scenes; no brands; no text.",
    "bostonvacay.com": "Photorealistic New England/Boston; brownstones, harbor, historic streets; no brands; no text.",
    "ipetzo.com":      "Photorealistic pet lifestyle; dogs/cats with owners outdoors/indoors; neutral interiors/parks; no visible logos or text.",
    "1-800deals.com":  "Photorealistic retail/ecommerce visuals; shopping scenes, parcels; generic products; clean backgrounds; no brands or text.",
}

# ------------------------- Helpers ------------------------- #

def slugify(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[’'`]", "", t)
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t or "image"

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

def to_webp_bytes(img: Image.Image, w: int, h: int, quality: int) -> bytes:
    img = img.convert("RGB")
    img = crop_to_aspect(img, w, h).resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=quality, method=6)
    return buf.getvalue()

def season_hint(keyword: str, site: str) -> str:
    k = keyword.lower()
    if any(x in k for x in ["winter","december","january","february","christmas","ski"]):
        return "winter conditions, snow present on ski runs and rooftops"
    if any(x in k for x in ["summer","june","july","august"]):
        return "summer conditions, lush green"
    if any(x in k for x in ["fall","autumn","september","october","november"]):
        return "autumn foliage"
    if any(x in k for x in ["spring","march","april","may"]):
        if "vail" in k or "vail" in site:
            return "late winter/early spring in the mountains; snow remains on ski runs"
        return "springtime"
    return "seasonally appropriate conditions"

def build_prompt(site: str, keyword: str, season_on: bool, inspiration: Optional[Dict] = None) -> str:
    base = SITE_PROFILES.get(site, SITE_PROFILES["vailvacay.com"])
    style_hints = []
    k = keyword.lower()

    if any(x in k for x in ["how far","directions","from","between"]):
        style_hints.append("scenic travel approach or roadway vantage; safe roadside view")
    if any(x in k for x in ["back bowls","ski","lift","gondola"]):
        style_hints.append("wide alpine vista that matches topic")
    if any(x in k for x in ["hotel","where to stay","resort"]):
        style_hints.append("inviting lodging exterior/interior, warm light")
    if "cigar" in k:
        style_hints.append("cozy lounge interior, warm ambient light, no branding")
    if any(x in k for x in ["don't ski","non-skiers","baby","kid","with kids","indoor"]):
        style_hints.append("family-friendly vibe, gentle activities")
    if any(x in k for x in ["county","river","mountain range","history","when did","when was"]):
        style_hints.append("editorial/documentary feel emphasizing place over people")
    if "ski patrol" in k:
        style_hints.append("winter slopes visible; snow present (avoid green summer grass)")

    season = season_hint(keyword, site) if season_on else "seasonally appropriate"
    style = ", ".join([s for s in style_hints if s]) or "scene appropriate to the topic"

    prompt = (f"{base} Balanced composition; natural light; editorial stock-photo feel. "
              f"Create an image for the topic: '{keyword}'. Landscape orientation. No words or typography. "
              f"Ensure {season}. Scene intent: {style}.")

    # Inspiration mode: guide an original composition without copying
    if inspiration:
        title = inspiration.get("title","reference image")
        notes = inspiration.get("notes","")
        prompt += (" Create a completely original scene inspired by the following cues from a reference photo "
                   f"(do not copy the exact building or layout; avoid any logos, signage, or readable text): "
                   f"Title: {title}. Key visual elements: {notes}.")
    return prompt

def openai_generate_image(prompt: str, size: str, api_key: str) -> Image.Image:
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "gpt-image-1", "prompt": prompt, "size": size}
    r = requests.post(url, headers=headers, json=payload, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
    data0 = r.json().get("data", [{}])[0]
    img_bytes = None
    if "b64_json" in data0:
        img_bytes = base64.b64decode(data0["b64_json"])
    elif "url" in data0 and data0["url"]:
        getr = requests.get(data0["url"], timeout=180)
        getr.raise_for_status()
        img_bytes = getr.content
    else:
        raise RuntimeError("OpenAI returned no image data.")
    return Image.open(io.BytesIO(img_bytes))

# -------------------- CC & Reference Search -------------------- #

def openverse_search(query: str, num: int = 4) -> List[Dict]:
    try:
        params = {"q": query, "license_type": "commercial", "page": 1, "page_size": max(1, min(num, 20))}
        r = requests.get("https://api.openverse.engineering/v1/images/", params=params, timeout=30)
        if r.status_code != 200:
            return []
        out = []
        for it in r.json().get("results", []):
            full = it.get("url") or it.get("foreign_landing_url")
            thumb = it.get("thumbnail") or full
            if not full: 
                continue
            out.append({
                "title": it.get("title") or "Openverse image",
                "thumb_url": thumb, "full_url": full,
                "source": "Openverse (CC)", "license": it.get("license") or "cc"}
            )
        return out
    except Exception:
        return []

def serpapi_google_images(query: str, serp_key: str, num: int = 4) -> List[Dict]:
    if not serp_key:
        return []
    try:
        params = {"engine": "google_images", "q": query, "ijn": "0", "api_key": serp_key}
        r = requests.get("https://serpapi.com/search", params=params, timeout=40)
        if r.status_code != 200:
            return []
        imgs = r.json().get("images_results", []) or []
        out = []
        for it in imgs[:num]:
            thumb = it.get("thumbnail") or it.get("original")
            full = it.get("original") or it.get("thumbnail")
            if not full:
                continue
            out.append({
                "title": it.get("title") or "Google image",
                "thumb_url": thumb, "full_url": full,
                "source": "Google (non-CC)", "license": "Reference only (non-CC)"}
            )
        return out
    except Exception:
        return []

def build_cc_query(keyword: str) -> str:
    q = keyword.strip()
    # Nudge venue queries to exterior/storefront; don't append site name ever
    if "exterior" not in q.lower() and "storefront" not in q.lower():
        q += " exterior storefront"
    return q

def is_noncc(c: Dict) -> bool:
    return "non-cc" in c.get("source","").lower() or "reference" in c.get("license","").lower()

def aggregate_candidates(keyword: str,
                         want_sources: List[str],
                         serp_key: str,
                         num: int,
                         prefer_cc: bool,
                         allow_ref_noncc: bool) -> List[Dict]:
    q = build_cc_query(keyword)
    results: List[Dict] = []

    if prefer_cc and "Openverse" in want_sources:
        results += openverse_search(q, num=num)

    if not results and allow_ref_noncc and "Google (via SerpAPI)" in want_sources:
        results += serpapi_google_images(q, serp_key, num=num)

    if not prefer_cc and "Google (via SerpAPI)" in want_sources and not results:
        results += serpapi_google_images(q, serp_key, num=num)

    seen = set(); deduped = []
    for r in results:
        fu = r.get("full_url")
        if not fu or fu in seen:
            continue
        seen.add(fu); deduped.append(r)
    return deduped

def download_image(url: str) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=40)
        if r.status_code != 200: 
            return None
        return Image.open(io.BytesIO(r.content))
    except Exception:
        return None

def save_cc_candidate(cand: Dict, out_w: int, out_h: int, q: int) -> Optional[bytes]:
    if is_noncc(cand):  # never save non-CC
        return None
    full = cand.get("full_url")
    if not full:
        return None
    img = download_image(full)
    if not img:
        return None
    return to_webp_bytes(img, out_w, out_h, q)

def guess_inspiration_notes(title: str, keyword: str) -> str:
    tl = (title or "").lower()
    kl = (keyword or "").lower()
    notes = []
    if any(x in tl+kl for x in ["snow","winter","ski","christmas"]):
        notes.append("snow on ground and rooftops; winter ambiance")
    if any(x in tl for x in ["patio","terrace","outdoor"]):
        notes.append("cozy outdoor patio seating; umbrellas")
    if any(x in tl+kl for x in ["vail","alpine","mountain"]):
        notes.append("alpine village architecture; mountains in background")
    if "string" in tl or "lights" in tl:
        notes.append("warm string lights")
    notes.append("no logos, no readable signage, original composition")
    return "; ".join(notes)

# ----------------------------- UI ----------------------------- #

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Pick CC photos (saved) or use non-CC thumbnails as inspiration for an original AI render. No logos/text; no copying.")

with st.sidebar:
    site = st.selectbox("Site style", list(SITE_PROFILES.keys()), index=0)
    out_w = st.number_input("Output width", 600, 2400, DEFAULT_OUTPUT_W, step=50)
    out_h = st.number_input("Output height", 300, 1350, DEFAULT_OUTPUT_H, step=25)
    webp_q = st.slider("WebP quality", 60, 95, DEFAULT_WEBP_QUALITY)

    st.markdown("### Generation")
    size = st.selectbox("OpenAI render size", OPENAI_IMG_SIZES, index=0)
    season_on = st.checkbox("Season-aware prompts", value=True)

    st.markdown("### API Keys")
    openai_key = st.text_input("OpenAI API key", type="password")
    serp_key = st.text_input("SerpAPI key (Google thumbnails)", type="password")

    st.markdown("### Prefer Real Photos?")
    prefer_cc = st.checkbox("Try Creative-Commons photo first", value=True)
    allow_ref_noncc = st.checkbox("If no CC found, show non-CC thumbnails for reference only", value=True, disabled=not prefer_cc)

    st.markdown("### CC selection")
    cc_mode = st.selectbox("Thumbnail picking", ["Manual pick (thumbnails)"])

    st.markdown("### Sources to search")
    sources = st.multiselect("Sources", ["Google (via SerpAPI)", "Openverse"], default=["Google (via SerpAPI)", "Openverse"])

    cand_per_kw = st.slider("Candidates per keyword", 1, 8, 4)

st.markdown("#### Keywords (one per line)")
kw_text = st.text_area("", height=150, placeholder="Tavern on the Square, Vail Colorado\nBlue Moose Pizza in Vail Colorado\nThings to do in Vail in October")

cols = st.columns([1,1])
do_gen = cols[0].button("Generate")
if cols[1].button("Clear"):
    st.session_state.clear()
    st.experimental_rerun()

if do_gen:
    keywords = [ln.strip() for ln in kw_text.splitlines() if ln.strip()]
    if not keywords:
        st.warning("Please paste at least one keyword.")
        st.stop()

    zip_buf = io.BytesIO()
    zf = zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED)
    previews: List[Tuple[str, bytes]] = []

    for idx, kw in enumerate(keywords, start=1):
        st.info(f"Working {idx}/{len(keywords)}: {kw}")
        exp = st.expander(f"📷  CC Thumbnails — pick one or use 'Inspiration' · {kw}", expanded=True)

        # Gather candidates
        cands = aggregate_candidates(
            keyword=kw,
            want_sources=sources,
            serp_key=serp_key,
            num=cand_per_kw,
            prefer_cc=prefer_cc,
            allow_ref_noncc=allow_ref_noncc
        )

        chosen_bytes: Optional[bytes] = None
        inspiration_choice: Optional[Dict] = None

        with exp:
            picked_idx = None
            noncc_options = [("None (no inspiration)", None)]

            if cands:
                for i, c in enumerate(cands):
                    thumb = c.get("thumb_url") or c.get("full_url")
                    if not thumb:
                        continue
                    col1, col2 = st.columns([1,2])
                    with col1:
                        st.image(thumb, caption=f"{c.get('source','')}", use_container_width=True)
                    with col2:
                        st.write(f"**{c.get('title','Image')}**")
                        st.write(f"License: {c.get('license','n/a')}")
                        st.write(f"Source: {c.get('source','')}")
                    st.markdown("---")

                # Radio for CC pick
                cc_labels = ["Use AI render"]
                cc_map = {}
                for i, c in enumerate(cands):
                    if not is_noncc(c):  # CC only for saving
                        label = f"{i+1}. {c.get('title','Image')} ({c.get('source','')})"
                        cc_labels.append(label); cc_map[label] = i
                chosen_label = st.radio(
                    "Pick a Creative-Commons thumbnail to save (or leave 'Use AI render'). Non-CC items below can be used as inspiration only.",
                    cc_labels, index=0
                )
                if chosen_label != "Use AI render":
                    picked_idx = cc_map.get(chosen_label, None)

                # Inspiration select (non-CC only)
                for i, c in enumerate(cands):
                    if is_noncc(c):
                        noncc_options.append((f"{i+1}. {c.get('title','Image')} (ref only)", c))
                opt_labels = [o[0] for o in noncc_options]
                insp_sel = st.selectbox("Inspiration (optional, non-CC reference):", opt_labels, index=0)
                sel_idx = opt_labels.index(insp_sel)
                selected_ref = noncc_options[sel_idx][1]

                insp_notes = ""
                if selected_ref:
                    default_notes = guess_inspiration_notes(selected_ref.get("title",""), kw)
                    insp_notes = st.text_area("Inspiration notes (edit as needed)",
                                              value=default_notes,
                                              help="Describe visual elements to emulate. We’ll generate an original scene and avoid copying logos/signage.")

                # If CC picked, try to save it now
                if picked_idx is not None:
                    cand = cands[picked_idx]
                    webp = save_cc_candidate(cand, out_w, out_h, webp_q)
                    if webp:
                        fname = f"{slugify(kw)}.webp"
                        zf.writestr(fname, webp)
                        previews.append((fname, webp))
                        chosen_bytes = webp
                        st.success("Saved Creative-Commons image.")
                    else:
                        st.warning("That candidate is non-CC or failed to download; will use AI.")

                # Prepare inspiration payload (used only if we do AI)
                if selected_ref:
                    inspiration_choice = {
                        "title": selected_ref.get("title","reference image"),
                        "notes": insp_notes or guess_inspiration_notes(selected_ref.get("title",""), kw)
                    }

            # If nothing chosen/saved, do AI render (optionally with inspiration)
            if chosen_bytes is None:
                if not openai_key:
                    st.error("OpenAI key missing and no CC image saved. Skipping this keyword.")
                else:
                    try:
                        prompt = build_prompt(site, kw, season_on, inspiration=inspiration_choice)
                        ai_img = openai_generate_image(prompt, size, openai_key)
                        webp = to_webp_bytes(ai_img, out_w, out_h, webp_q)
                        fname = f"{slugify(kw)}.webp"
                        zf.writestr(fname, webp)
                        previews.append((fname, webp))
                        st.success("AI render saved.")
                    except Exception as e:
                        st.error(f"OpenAI generation failed: {e}")

        st.markdown("---")

    zf.close(); zip_buf.seek(0)

    if previews:
        st.success("Done! Download your images below.")
        st.download_button("⬇️ Download ZIP", data=zip_buf,
                           file_name=f"{slugify(site)}_images.zip", mime="application/zip")
        st.markdown("### Previews & individual downloads")
        cols = st.columns(3)
        for i, (fname, data_bytes) in enumerate(previews):
            with cols[i % 3]:
                st.image(data_bytes, caption=fname, use_container_width=True)
                st.download_button("Download", data=data_bytes, file_name=fname, mime="image/webp")
    else:
        st.warning("Nothing was generated. Try different keywords or check API keys.")
