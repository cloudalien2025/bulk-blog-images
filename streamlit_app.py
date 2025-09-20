# ImageForge v0.9.4 — Bulk blog/Pinterest image generator with optional CC photo picker
# Requirements: streamlit, requests, pillow
# APIs: OpenAI (required), SerpAPI (optional). Openverse needs no key.

import base64, io, os, re, json, time, random, datetime as dt
from typing import List, Dict, Optional, Tuple

import requests
from PIL import Image
import streamlit as st


# ---------------------------
# App config
# ---------------------------
APP_NAME = "ImageForge"
APP_VERSION = "v0.9.4"
DEFAULT_QUALITY = 82
BLOG_W, BLOG_H = 1200, 675           # 16:9
PIN_W, PIN_H  = 1000, 1500           # 2:3 (Pinterest standard pin)

# Render (OpenAI fixed sizes) -> (our crop size)
RENDER_MAP = {
    "Blog 1200×675 (16:9)": ("1536x1024", BLOG_W, BLOG_H),   # landscape
    "Pinterest 1000×1500 (2:3)": ("1024x1536", PIN_W, PIN_H) # portrait
}

SITE_PROFILES = {
    "vailvacay.com":  "Colorado Rockies alpine resort vibe: slopes, aspens, evergreen forests, Gore Creek, Bavarian village details.",
    "bostonvacay.com": "New England coastal/city feel: brick townhouses, harbor, brownstones, cobblestones, seafood scene.",
    "bangkokvacay.com": "Urban SE Asia: neon streets, temples, markets, canals, rooftop bars, tropical light.",
    "ipetzo.com":      "Pet lifestyle: happy dogs/cats with owners indoors/outdoors; soft light; no brand logos.",
    "1-800deals.com":  "Retail/e-commerce feel: shopping scenes, parcels, generic products on clean backgrounds; no brands."
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

def crop_to_aspect(img: Image.Image, w: int, h: int) -> Image.Image:
    target = w / h
    iw, ih = img.size
    cur = iw / ih
    if cur > target:
        new_w = int(ih * target); left = (iw - new_w)//2
        box = (left, 0, left+new_w, ih)
    else:
        new_h = int(iw / target); top = (ih - new_h)//2
        box = (0, top, iw, top+new_h)
    return img.crop(box)

def to_webp_bytes(img: Image.Image, w: int, h: int, quality: int) -> bytes:
    img = img.convert("RGB")
    img = crop_to_aspect(img, w, h).resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=int(quality), method=6)
    return buf.getvalue()

def safe_download_image(url: str, timeout: int = 25) -> Optional[Image.Image]:
    try:
        if not url: 
            return None
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


# ---------------------------
# Season & content heuristics
# ---------------------------
MONTH_TO_SEASON = {
    1:"winter",2:"winter",3:"winter",4:"spring",5:"spring",6:"summer",
    7:"summer",8:"summer",9:"fall",10:"fall",11:"winter",12:"winter"
}
MONTH_WORDS = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12
}

def detect_month_from_text(text: str) -> Optional[int]:
    t = text.lower()
    for mname, mnum in MONTH_WORDS.items():
        if mname in t:
            return mnum
    return None

def is_ski_term(t: str) -> bool:
    t = t.lower()
    keys = ["ski", "back bowl", "powder", "lift ticket", "patrol", "gondola", "snowboard", "glades"]
    return any(k in t for k in keys)

def is_indoor_term(t: str) -> bool:
    t = t.lower()
    keys = ["indoor", "museum", "cigar lounge", "spa", "aquarium", "shopping mall", "arcade", "escape room"]
    return any(k in t for k in keys)

def season_for_site(site: str, keyword: str) -> str:
    m = detect_month_from_text(keyword)
    now_month = dt.datetime.now().month
    month = m or now_month
    season = MONTH_TO_SEASON.get(month, "any")

    if site == "vailvacay.com":
        if is_ski_term(keyword) and month in (11,12,1,2,3,4):
            return "winter"
        if "october" in keyword.lower():
            return "fall"
    return season

def style_hints(keyword: str, site: str, season: str) -> str:
    k = keyword.lower()
    hints = []
    if is_indoor_term(k):
        hints.append("cozy indoor scene; natural window light; shallow depth of field")
    if "cigar" in k:
        hints.append("upscale lounge interior; leather club chairs; no visible brands")
    if any(x in k for x in ["where to stay", "hotel review", "hotel", "resort"]):
        hints.append("inviting lodge exterior/interior; warm dusk or golden-hour light")
    if any(x in k for x in ["how far","drive","directions","route","between","to vail from"]):
        hints.append("scenic roadway or approach view; safe roadside vantage; seasonal landscape")
    if any(x in k for x in ["best burger","seafood","restaurant","coffee","bar"]):
        hints.append("editorial food/venue vibe; subtle human presence; no legible menu prices")

    if season == "winter":
        hints.append("fresh snow; winter clothing; puffs of breath; low warm sun")
    elif season == "fall":
        hints.append("golden aspens and evergreens" if site=="vailvacay.com" else "rich fall foliage")
    elif season == "summer":
        hints.append("lush summer greens; bright airy feel")
    elif season == "spring":
        hints.append("early thaw; bright sky; lingering snow patches optional")

    return ", ".join(hints) if hints else "scene appropriate to the topic"

def build_prompt(site: str, keyword: str, season: str, orientation: str) -> str:
    base = SITE_PROFILES.get(site, SITE_PROFILES[DEFAULT_SITE])
    style = style_hints(keyword, site, season)
    factual_guard = "avoid readable prices or brand names; if any signage appears keep it generic and minimal"
    return (
        f"Photorealistic travel blog image. {base} "
        f"Subject: an image that illustrates the topic: '{keyword}'. "
        f"Seasonal context: {season}. Orientation: {orientation}. "
        f"Balanced composition, editorial stock-photo feel, natural light, minimal processing. "
        f"Scene intent: {style}. No words or typography overlay; {factual_guard}."
    )


# ---------------------------
# OpenAI images with compatibility fallback
# ---------------------------
def openai_image_bytes(prompt: str, size: str, api_key: str) -> bytes:
    """
    Try gpt-image-1 with b64_json; if API returns 'unknown parameter: response_format',
    retry without response_format and download by URL.
    """
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Attempt 1: request base64
    payload = {"model": "gpt-image-1", "prompt": prompt, "size": size, "response_format": "b64_json"}
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if r.status_code == 200:
        b64 = r.json()["data"][0]["b64_json"]
        return base64.b64decode(b64)

    # If response_format not supported, try URL flow
    try:
        err = r.json()
        msg = err.get("error", {}).get("message", "").lower()
    except Exception:
        msg = ""
    if "response_format" in msg or r.status_code == 400:
        payload2 = {"model": "gpt-image-1", "prompt": prompt, "size": size}
        r2 = requests.post(url, headers=headers, json=payload2, timeout=120)
        if r2.status_code != 200:
            raise RuntimeError(f"OpenAI error {r2.status_code}: {r2.text}")
        img_url = r2.json()["data"][0]["url"]
        img = safe_download_image(img_url)
        if not img:
            raise RuntimeError("Failed to download image from OpenAI URL.")
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()

    # Otherwise, bubble up original error
    raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")


# ---------------------------
# LSI variants
# ---------------------------
def heuristic_lsi(keyword: str, k: int) -> List[str]:
    bases = [
        "guide", "map", "tips", "with kids", "on a budget", "date ideas",
        "insider", "must-see", "hidden gems", "first-timers"
    ]
    random.shuffle(bases)
    out = []
    for i in range(k):
        if i < len(bases):
            out.append(f"{keyword} — {bases[i]}")
        else:
            out.append(f"{keyword} ideas #{i+1}")
    return out

def openai_lsi(keyword: str, k: int, api_key: str) -> List[str]:
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        prompt = (
            "Generate concise LSI subtopics (4-6 words each) for this travel-blog keyword. "
            "No punctuation except spaces; no quotes; return as JSON array of strings. "
            f"Keyword: {keyword}. Count: {k}."
        )
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={"model":"gpt-4o-mini","messages":[{"role":"user","content":prompt}],"temperature":0.7},
            timeout=60
        )
        if r.status_code != 200:
            return heuristic_lsi(keyword, k)
        txt = r.json()["choices"][0]["message"]["content"]
        arr = json.loads(txt) if txt.strip().startswith("[") else heuristic_lsi(keyword, k)
        return [str(x) for x in arr][:k]
    except Exception:
        return heuristic_lsi(keyword, k)


# ---------------------------
# Reference image search (SerpAPI + Openverse)
# ---------------------------
def serpapi_google_images(query: str, serp_key: str, num: int = 6, cc_only: bool = False) -> List[Dict]:
    if not serp_key:
        return []
    try:
        params = {"engine": "google_images", "q": query, "ijn": "0", "num": num, "api_key": serp_key}
        if cc_only:
            params["tbs"] = "sur:cl"  # CC filter
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=60)
        if r.status_code != 200:
            return []
        data = r.json()
        results = []
        for it in data.get("images_results", [])[:num]:
            results.append({
                "thumb_url": it.get("thumbnail"),
                "original_url": it.get("original"),
                "title": it.get("title"),
                "host": it.get("source"),
                "owner_name": "",
                "license": "CC (as reported by Google)" if cc_only else "Unspecified",
                "license_url": "",
                "attribution_url": it.get("link"),
                "source": "GoogleCC" if cc_only else "Google",
            })
        return results
    except Exception:
        return []

def openverse_search(query: str, num: int = 6) -> List[Dict]:
    try:
        r = requests.get(
            "https://api.openverse.engineering/v1/images/",
            params={"q":query, "page_size":num},
            timeout=60
        )
        if r.status_code != 200:
            return []
        out = []
        for item in r.json().get("results", [])[:num]:
            out.append({
                "thumb_url": item.get("thumbnail"),
                "original_url": item.get("url"),
                "title": item.get("title"),
                "owner_name": item.get("creator"),
                "license": item.get("license"),
                "license_url": item.get("license_url"),
                "attribution_url": item.get("foreign_landing_url"),
                "source": "Openverse",
            })
        return out
    except Exception:
        return []

def aggregate_cc_candidates(keyword: str, site: str, want_sources: List[str], serp_key: str, num: int) -> List[Dict]:
    q = f"{keyword} {site.split('.')[0].replace('-', ' ')}"
    results = []
    if "Google (via SerpAPI)" in want_sources:
        results += serpapi_google_images(q, serp_key, num=num, cc_only=True)
    if "Openverse" in want_sources:
        results += openverse_search(q, num=num)
    dedup = {}
    for r in results:
        url = r.get("original_url")
        if url and (url not in dedup):
            dedup[url] = r
    return list(dedup.values())[:num]


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title=f"{APP_NAME} {APP_VERSION}", layout="wide")
st.title(f"{APP_NAME} {APP_VERSION}")
st.caption("Bulk blog & Pinterest images — AI renders with optional CC photo picker (+ season-aware prompts).")

with st.sidebar:
    st.subheader("🔑 API Keys")
    openai_key = st.text_input("OpenAI API key *", type="password", help="Required")
    serp_key = st.text_input("SerpAPI key (optional)", type="password", help="For Google Images search (CC)")

    st.subheader("🖼️ Output")
    render_choice = st.selectbox("Size / orientation", list(RENDER_MAP.keys()), index=0)
    dalle_size, OUT_W, OUT_H = RENDER_MAP[render_choice]
    quality = st.slider("WebP quality", 60, 95, DEFAULT_QUALITY)

    st.subheader("🧩 Variations")
    imgs_per_kw = st.slider("Images per keyword", 1, 10, 1, help="Use LSI to expand if >1")
    lsi_mode = st.selectbox("LSI method", ["Heuristic", "AI (OpenAI)"], index=0)

    st.subheader("📸 Prefer Real Photos?")
    prefer_cc = st.checkbox("Try Creative-Commons photo first (if available)", value=False)
    cc_mode = st.selectbox("CC selection", ["Hands-free (auto pick)", "Manual pick (thumbnails)"], index=1, disabled=not prefer_cc)
    cc_sources = st.multiselect("Sources to search", ["Google (via SerpAPI)", "Openverse"],
                                default=["Google (via SerpAPI)", "Openverse"], disabled=not prefer_cc)
    cc_candidates = st.slider("Candidates per keyword", 0, 8, 4, disabled=not prefer_cc)

    st.subheader("⚙️ Advanced")
    season_aware = st.checkbox("Season-aware prompts", value=True)
    st.caption("OpenAI renders follow allowed sizes: 1024×1024, 1024×1536, 1536×1024. We crop to your target size.")

keywords_text = st.text_area(
    "Keywords (one per line)",
    height=240,
    placeholder="Things to Do in Vail in October\nBest seafood restaurant in Boston\nIndoor activities Vail"
)

colA, colB = st.columns([1,1])
run_btn = colA.button("Generate")
clear_btn = colB.button("Clear")
if clear_btn:
    st.session_state.clear()
    st.experimental_rerun()

if run_btn:
    if not openai_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    keywords = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
    if not keywords:
        st.error("Please paste at least one keyword.")
        st.stop()

    work_items: List[Tuple[str,str,str]] = []
    for kw in keywords:
        if imgs_per_kw == 1:
            work_items.append((kw, kw, slugify(kw)))
        else:
            if lsi_mode.startswith("AI"):
                variants = openai_lsi(kw, imgs_per_kw - 1, openai_key)
            else:
                variants = heuristic_lsi(kw, imgs_per_kw - 1)
            all_kws = [kw] + variants
            for i, v in enumerate(all_kws, 1):
                lab = f"{kw} — v{i}" if i > 1 else kw
                work_items.append((kw, v, slugify(v)))

    prog = st.progress(0.0)
    status = st.empty()
    previews: List[Tuple[str, bytes]] = []
    attributions: List[str] = []
    meta: List[str] = []

    done = 0
    total = len(work_items)

    manual_selections: Dict[int, int] = {}
    cc_cache: Dict[int, List[Dict]] = {}

    if prefer_cc and cc_mode.startswith("Manual"):
        st.subheader("📸 CC Thumbnails — choose one or leave 'AI render'")
        for idx, (orig_kw, actual_kw, fname_base) in enumerate(work_items):
            cands = aggregate_cc_candidates(
                actual_kw, DEFAULT_SITE, cc_sources, serp_key, cc_candidates or 0
            ) if cc_candidates else []
            cc_cache[idx] = cands
            with st.expander(f"{actual_kw}"):
                if not cands:
                    st.info("No CC candidates found for this one. It will use AI.")
                else:
                    cols = st.columns(min(4, len(cands)))
                    for i, c in enumerate(cands):
                        with cols[i % 4]:
                            st.image(c.get("thumb_url") or c.get("original_url"), use_container_width=True)
                            st.caption(f"{c.get('source','')} — {c.get('title','')[:40]}")
                    pick = st.selectbox(
                        "Pick a CC photo (or choose AI render):",
                        ["AI render"] + [f"Use candidate #{i+1}" for i in range(len(cands))],
                        key=f"ccpick_{idx}"
                    )
                    if pick != "AI render":
                        manual_selections[idx] = int(pick.split("#")[-1]) - 1
        st.write("---")

    def save_cc(cand: Dict, fname_base: str) -> Optional[Tuple[bytes, str]]:
        img = safe_download_image(cand.get("original_url", ""))
        if not img:
            return None
        webp_local = to_webp_bytes(img, OUT_W, OUT_H, quality)
        src = cand.get("source", "")
        if src == "Openverse":
            line = (
                f"{fname_base} — Openverse: \"{cand.get('title','')}\" by {cand.get('owner_name','')} "
                f"({cand.get('license','')}) {cand.get('license_url','')} | {cand.get('attribution_url','')}"
            )
        else:  # GoogleCC
            line = (
                f"{fname_base} — Google Images (CC as reported): {cand.get('original_url','')} | "
                f"Title: {cand.get('title','')} | Source: {cand.get('host','')}"
            )
        return webp_local, line

    for idx, (orig_kw, actual_kw, fname_base) in enumerate(work_items):
        status.info(f"Working {idx+1}/{total}: {actual_kw}")
        used_real = False
        webp_bytes: Optional[bytes] = None

        if prefer_cc and cc_mode.startswith("Manual") and idx in manual_selections:
            cands = cc_cache.get(idx, [])
            pick_i = manual_selections[idx]
            if 0 <= pick_i < len(cands):
                res = save_cc(cands[pick_i], fname_base)
                if res:
                    webp_bytes, attr = res
                    used_real = True
                    attributions.append(attr)
                    meta.append(f"{actual_kw} → CC photo (manual)")
                else:
                    meta.append(f"{actual_kw} → chosen CC failed; switched to AI")

        if prefer_cc and cc_mode.startswith("Hands-free") and not used_real and cc_candidates:
            auto_cands = aggregate_cc_candidates(
                actual_kw, DEFAULT_SITE, cc_sources, serp_key, cc_candidates
            )
            if auto_cands:
                res = save_cc(auto_cands[0], fname_base)
                if res:
                    webp_bytes, attr = res
                    used_real = True
                    attributions.append(attr)
                    meta.append(f"{actual_kw} → CC photo (auto)")

        if not used_real:
            site = DEFAULT_SITE
            season = season_for_site(site, actual_kw) if season_aware else "any"
            orientation = "landscape" if OUT_W >= OUT_H else "portrait"
            prompt = build_prompt(site, actual_kw, season, orientation)
            try:
                png = openai_image_bytes(prompt, dalle_size, openai_key)
                img = Image.open(io.BytesIO(png)).convert("RGB")
                webp_bytes = to_webp_bytes(img, OUT_W, OUT_H, quality)
                attributions.append(f"{fname_base} — AI (OpenAI gpt-image-1). Prompt season='{season}'.")
                meta.append(f"{actual_kw} → AI render")
            except Exception as e:
                st.error(f"{actual_kw}: {e}")
                meta.append(f"{actual_kw} → ERROR: {e}")
                continue

        if webp_bytes:
            previews.append((f"{slugify(actual_kw)}.webp", webp_bytes))

        st.session_state["site"] = DEFAULT_SITE
        prog.progress((idx+1)/total)

    if previews:
        import zipfile
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
            for fname, data in previews:
                z.writestr(fname, data)
            if attributions:
                z.writestr("attributions.txt", "\n".join(attributions))
            if meta:
                z.writestr("run_log.txt", "\n".join(meta))
        zip_buf.seek(0)

        st.success("Done! Download your images below.")
        st.download_button("⬇️ Download ZIP", data=zip_buf, file_name=f"{slugify(DEFAULT_SITE)}_{int(time.time())}.zip", mime="application/zip")

        st.subheader("Previews")
        cols = st.columns(3)
        for i, (fname, data) in enumerate(previews):
            with cols[i % 3]:
                st.image(data, caption=fname, use_container_width=True)
                st.download_button("Download", data=data, file_name=fname, mime="image/webp", key=f"dlf_{i}")

        if attributions:
            st.subheader("Attributions / Notes")
            st.code("\n".join(attributions), language="text")
    else:
        st.warning("Nothing was generated. Try different keywords or check API keys.")
