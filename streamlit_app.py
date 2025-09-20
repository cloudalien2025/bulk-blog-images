# ImageForge v0.8 "Explorer"
# Adds SerpAPI thumbnail source + LSI expansion (Off / Heuristic / OpenAI)
# Keeps Reference-Lock, Fidelity/retry, Venue mode, Pinterest, and CC-first logic.

import os, io, re, time, base64, zipfile, json, random, math
from typing import List, Dict, Optional, Tuple

import requests
from PIL import Image, ImageStat
import streamlit as st

APP_TITLE = "ImageForge v0.8 — Explorer (SerpAPI + LSI)"

# ---------- OUTPUT SIZES ----------
DEFAULT_OUTPUT_W, DEFAULT_OUTPUT_H = 1200, 675
PIN_OUTPUT_W, PIN_OUTPUT_H = 1000, 1500
OPENAI_SIZES = ["1536x1024", "1024x1536", "1024x1024"]

# ---------- SITE PROFILES ----------
SITE_PROFILES = {
    "vailvacay.com":  "Colorado Rockies resort/village, alpine architecture, fir/spruce, riverside paths, ski terrain; photoreal; editorial stock feel; no text/logos.",
    "bostonvacay.com":"Historic New England city, brick & brownstone, harbor/waterfront, parks; photoreal; editorial stock feel; no text/logos.",
    "bangkokvacay.com":"Southeast Asian metropolis, street food/night markets, temples/rooftops/skyline; photoreal; editorial stock feel; no text/logos.",
    "ipetzo.com":      "Pet lifestyle scenes, owners with dogs/cats in parks/neutral interiors; photoreal; editorial stock feel; no brands/text.",
    "1-800deals.com":  "Generic shopping/ecommerce scenes, parcels, aisles; clean backgrounds; photoreal; no brands/text."
}
DEFAULT_SITE = "vailvacay.com"

# ---------- UTILS ----------
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
        return img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio); top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))

def save_webp_bytes(img: Image.Image, w: int, h: int, q: int) -> bytes:
    img = img.convert("RGB")
    img = crop_to_aspect(img, w, h).resize((w, h), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, "WEBP", quality=q, method=6)
    return buf.getvalue()

def fetch_image_bytes(url: str, timeout: int = 20) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        pass
    return None

def luminance_fraction(img: Image.Image, lo: int = 220) -> float:
    g = img.convert("L"); hist = g.histogram()
    total = sum(hist); white = sum(hist[lo:])
    return (white/total) if total else 0.0

def color_presence(img: Image.Image, hue_range: Tuple[int,int], sat_lo=80, val_lo=70) -> float:
    import colorsys
    img = img.convert("RGB").resize((256,256))
    w,h = img.size; count=0; hit=0; h1,h2 = hue_range; wrap = h2 < h1
    for y in range(h):
        for x in range(w):
            r,g,b = img.getpixel((x,y)); r/=255; g/=255; b/=255
            H,S,V = colorsys.rgb_to_hsv(r,g,b)
            Hdeg, S100, V100 = int(H*360), int(S*100), int(V*100)
            count += 1
            cond_h = (Hdeg>=h1 or Hdeg<=h2) if wrap else (h1<=Hdeg<=h2)
            if cond_h and S100>=sat_lo and V100>=val_lo: hit += 1
    return hit/max(1,count)

def analyze_reference(url: str) -> Dict:
    data = fetch_image_bytes(url)
    if not data: return {"ok": False}
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return {"ok": False}

    snow_frac = luminance_fraction(img, lo=225)
    yellow_frac = color_presence(img, (35,60), 70, 65)
    blue_frac = color_presence(img, (195,240), 40, 55)
    stat = ImageStat.Stat(img); mean = stat.mean
    warm_hint = (mean[0] > mean[2])

    checklist = set()
    if snow_frac >= 0.06: checklist.add("snow_on_ground_and_roofs")
    if yellow_frac >= 0.015: checklist.add("patio_umbrellas_or_warm_yellow_accents")
    checklist.add("alpine_gabled_facade_with_balconies")
    if blue_frac >= 0.04: checklist.add("visible_sky_background")

    return {"ok": True, "snow_frac": snow_frac, "yellow_frac": yellow_frac,
            "blue_frac": blue_frac, "warm_hint": warm_hint, "checklist": list(checklist)}

def expectation_from_features(feat: Dict) -> Dict:
    return {
        "expect_snow": "snow_on_ground_and_roofs" in feat.get("checklist", []),
        "expect_yellow": "patio_umbrellas_or_warm_yellow_accents" in feat.get("checklist", [])
    }

def postcheck(img_bytes: bytes, expect: Dict) -> bool:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return False
    ok = True
    if expect.get("expect_snow") and luminance_fraction(img, lo=225) < 0.04: ok = False
    if expect.get("expect_yellow") and color_presence(img, (35,60), 70,65) < 0.008: ok = False
    return ok

# ---------- SEARCH SOURCES ----------
def openverse_search(q: str, n: int = 4) -> List[Dict]:
    url = "https://api.openverse.engineering/v1/images/"
    params = {"q": q, "license": "cc0,cc-by,cc-by-sa,cc-by-nd", "page_size": min(10, max(1, n))}
    try:
        r = requests.get(url, params=params, timeout=20)
        out = []
        if r.status_code == 200:
            for it in r.json().get("results", []):
                link = it.get("url") or it.get("foreign_landing_url")
                thumb = it.get("thumbnail") or link
                if link:
                    out.append({"title": it.get("title") or "", "url": link,
                                "thumb": thumb, "source":"Openverse (CC)",
                                "license": it.get("license") or "cc", "is_cc": True})
                if len(out) >= n: break
        return out
    except Exception:
        return []

def google_cse_images(q: str, api_key: str, cx: str, n: int = 4) -> List[Dict]:
    if not api_key or not cx: return []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cx, "q": q, "searchType":"image",
              "num": min(10, max(1,n)), "safe":"active", "imgSize":"large"}
    try:
        r = requests.get(url, params=params, timeout=20)
        out = []
        if r.status_code == 200:
            for it in r.json().get("items", []):
                link = it.get("link")
                thumb = (it.get("image") or {}).get("thumbnailLink") or link
                if link:
                    out.append({"title": it.get("title") or "", "url": link,
                                "thumb": thumb, "source":"Google (non-CC)",
                                "license":"reference-only", "is_cc": False})
                if len(out) >= n: break
        return out
    except Exception:
        return []

def serpapi_images(q: str, serp_key: str, n: int = 4) -> List[Dict]:
    if not serp_key: return []
    url = "https://serpapi.com/search.json"
    params = {"engine":"google", "q": q, "tbm":"isch", "ijn":"0",
              "api_key": serp_key, "num": min(10, max(1,n))}
    try:
        r = requests.get(url, params=params, timeout=20)
        out = []
        if r.status_code == 200:
            for it in r.json().get("images_results", []):
                link = it.get("original") or it.get("thumbnail") or it.get("source")
                thumb = it.get("thumbnail") or link
                if link:
                    out.append({"title": it.get("title") or "", "url": link,
                                "thumb": thumb, "source":"SerpAPI (Google Images, non-CC)",
                                "license":"reference-only", "is_cc": False})
                if len(out) >= n: break
        return out
    except Exception:
        return []

# ---------- PROMPTS ----------
def venue_mode_hint(keyword: str) -> bool:
    k = keyword.lower()
    return any(x in k for x in ["restaurant","bar","cafe","coffee","hotel","inn","lodge","tavern","rooftop","resort"])

def detect_season_from_keyword(keyword: str) -> Optional[str]:
    k = keyword.lower()
    tags = [("january","winter"),("february","winter"),("march","late winter"),
            ("april","spring shoulder"),("may","spring"),("june","summer"),
            ("july","summer"),("august","summer"),("september","early fall"),
            ("october","fall"),("november","late fall"),("december","winter"),
            ("christmas","winter"),("snow","winter"),("ski","winter"),("back bowls","winter")]
    for name,tag in tags:
        if name in k: return tag
    return None

def build_prompt(site: str, keyword: str, season_mode: str,
                 ref_feat: Optional[Dict], fidelity: int,
                 lock_strength: str, venue: bool) -> str:
    base = SITE_PROFILES.get(site, SITE_PROFILES[DEFAULT_SITE])
    # season
    season_hint = ""
    if season_mode == "Auto":
        s = detect_season_from_keyword(keyword) or ""
        season_hint = f"Season cues: {s}. " if s else ""
    elif season_mode in ("Winter","Summer","Fall","Spring"):
        season_hint = f"Season cues: {season_mode}. "

    # constraints
    constraints = []
    if ref_feat and ref_feat.get("ok"):
        cl = set(ref_feat.get("checklist", []))
        names = {
            "alpine_gabled_facade_with_balconies":"alpine gabled façade with balconies",
            "snow_on_ground_and_roofs":"fresh snow on ground and roofs",
            "patio_umbrellas_or_warm_yellow_accents":"patio with umbrellas and warm yellow accents",
            "visible_sky_background":"sky and mountain backdrop",
        }
        for key in ["alpine_gabled_facade_with_balconies","snow_on_ground_and_roofs",
                    "patio_umbrellas_or_warm_yellow_accents","visible_sky_background"]:
            if key in cl: constraints.append(names[key])

    must = ("must clearly include" if fidelity >= 60 else "should include")
    lock_phrase = ""
    if lock_strength == "Strong" and constraints:
        lock_phrase = f" The scene {must}: " + ", ".join(constraints) + "."
    elif lock_strength == "Moderate" and constraints:
        lock_phrase = f" The scene should feature: " + ", ".join(constraints) + "."

    venue_bits = ""
    if venue:
        venue_bits = (" Exterior, eye-level view of the venue façade and patio seating if present; "
                      "warm window glow or string lights at dusk is welcome; "
                      "do not show readable signage or logos; keep everything generic.")

    strict = ""
    if fidelity >= 80:
        strict = (" Balanced yet tight composition mirroring the reference layout. "
                  "Prominent, unambiguous depiction of the listed elements.")
    elif fidelity >= 50:
        strict = " Keep composition similar to the reference vibe and include the listed elements."
    else:
        strict = " General vibe match is sufficient."

    return (f"{base} {season_hint}"
            f"Create a photorealistic, editorial-style stock image for: '{keyword}'. "
            f"Landscape orientation. No words, watermarks, trademarks, or readable storefront text. "
            f"{venue_bits}{lock_phrase}{strict}")

# ---------- OPENAI ----------
def dalle_generate_png_bytes(prompt: str, size: str, api_key: str) -> bytes:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model":"gpt-image-1", "prompt": prompt, "size": size, "response_format":"b64_json"}
    r = requests.post("https://api.openai.com/v1/images/generations",
                      headers=headers, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
    b64 = r.json()["data"][0]["b64_json"]
    return base64.b64decode(b64)

def openai_lsi_terms(keyword: str, k: int, api_key: str) -> List[str]:
    """Use OpenAI text to produce LSI-style subtopics. Returns at most k unique lines."""
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type":"application/json"}
        sys = ("You generate SEO LSI ideas. Return a plain list with one short phrase per line, "
               "no numbering, no quotes.")
        user = f"Main keyword: {keyword}\nReturn {k} closely-related subtopics suitable for distinct images."
        body = {"model":"gpt-4o-mini", "messages":[{"role":"system","content":sys},
                                                   {"role":"user","content":user}],
                "temperature":0.5}
        r = requests.post(url, headers=headers, json=body, timeout=60)
        if r.status_code != 200: return []
        txt = r.json()["choices"][0]["message"]["content"]
        lines = [ln.strip(" -•\t") for ln in txt.splitlines() if ln.strip()]
        uniq = []
        for ln in lines:
            if ln.lower() not in [x.lower() for x in uniq]:
                uniq.append(ln)
            if len(uniq) >= k: break
        return uniq[:k]
    except Exception:
        return []

def heuristic_lsi(keyword: str, k: int) -> List[str]:
    """Zero-cost LSI heuristic: mixes generic buckets with detected city/season terms."""
    k = max(0,k)
    out = []
    klow = keyword.lower()
    loc = ""
    m = re.search(r"in ([a-z ,\-]+)$", klow) or re.search(r"in ([a-z ,\-]+)\b", klow)
    if m: loc = m.group(1).strip(", ").title()

    buckets = ["best photo spots", "scenic viewpoint", "cozy cafés", "family-friendly activity",
               "hidden gem", "budget-friendly idea", "romantic spot", "day trip", "local market",
               "popular hike", "waterfront walk", "museum/gallery", "night view", "sunrise/sunset spot"]
    random.shuffle(buckets)
    for b in buckets:
        item = f"{b} {('in ' + loc) if loc else ''}".strip()
        if item.lower() != keyword.lower():
            out.append(item)
        if len(out) >= k: break
    return out[:k]

# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Keys")
    openai_key = st.text_input("OpenAI API key", type="password")
    g_key = st.text_input("Google CSE API key (for thumbnails, optional)", type="password")
    g_cx  = st.text_input("Google CSE CX (engine id, optional)", type="password")
    serp_key = st.text_input("SerpAPI key (optional)", type="password")

    st.subheader("Output")
    site = st.selectbox("Site style", list(SITE_PROFILES.keys()),
                        index=list(SITE_PROFILES.keys()).index(DEFAULT_SITE))
    render_size = st.selectbox("Render base size", OPENAI_SIZES, index=0)
    webp_quality = st.slider("WebP quality", 60, 95, 82)
    make_pins = st.checkbox("Also make a Pinterest image (1000×1500)", value=False)

    st.subheader("Thumbnail sources")
    use_openverse = st.checkbox("Openverse (CC)", value=True)
    use_cse = st.checkbox("Google CSE (non-CC reference)", value=True)
    use_serp = st.checkbox("SerpAPI (Google Images, non-CC reference)", value=True)
    prefer_cc = st.checkbox("CC-first", value=True)
    allow_noncc = st.checkbox("Allow non-CC reference if no CC found", value=True)
    cand_per_kw = st.slider("Candidates per keyword", 2, 8, 4)

    st.subheader("Reference Lock & Season")
    lock_strength = st.selectbox("Reference Lock", ["Strong","Moderate","Off"], index=0)
    fidelity = st.slider("Fidelity (strictness & retries)", 0, 100, 70)
    season_mode = st.selectbox("Season cues", ["Auto","Winter","Summer","Fall","Spring"])

    st.subheader("LSI expansion")
    lsi_mode = st.selectbox("LSI method", ["Off","Heuristic","OpenAI"], index=0)
    images_per_kw = st.slider("Images per keyword (with LSI)", 1, 10, 1)

st.caption("Paste keywords (one per line).")
keywords_text = st.text_area("Keywords", height=180,
    placeholder="Tavern on the Square, Vail Colorado\nBest seafood restaurant in Boston\nThings to do in Vail in October")

colA, colB = st.columns([1,1])
generate_btn = colA.button("Generate", type="primary")
if colB.button("Clear"): st.session_state.clear(); st.experimental_rerun()

# ---------- THUMBNAIL FLOW ----------
def collect_thumbnails(q: str, n: int) -> List[Dict]:
    items: List[Dict] = []
    # order: CC first if checked
    if prefer_cc and use_openverse:
        items.extend(openverse_search(q, n))
    # add non-CC if allowed and still short
    if allow_noncc:
        still = n - len(items)
        if still > 0 and use_cse:
            items.extend(google_cse_images(q, g_key, g_cx, still))
        still = n - len(items)
        if still > 0 and use_serp:
            items.extend(serpapi_images(q, serp_key, still))
    return items[:n]

def autopick_thumb(items: List[Dict]) -> Optional[Dict]:
    if not items: return None
    for it in items:
        if it.get("is_cc"): return it
    return items[0]

def show_thumb_picker(q: str, items: List[Dict]) -> Optional[Dict]:
    if not items:
        st.info("No candidates found. It will render with AI only.")
        return None
    pick = None
    for i, it in enumerate(items):
        c1,c2 = st.columns([3,5])
        with c1: st.image(it.get("thumb") or it["url"], use_container_width=True)
        with c2:
            st.write(f"**{it.get('title') or 'Untitled'}**")
            st.caption(f"Source: {it['source']} — license: {it['license']}")
            if st.button(f"Use this as reference (#{i+1})", key=f"use_{slugify(q)}_{i}"):
                pick = it
    return pick

# ---------- RUN ----------
if generate_btn:
    if not openai_key:
        st.warning("Please enter your OpenAI API key."); st.stop()
    keywords = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
    if not keywords:
        st.warning("Please paste at least one keyword."); st.stop()

    # LSI expansion plan
    def expand_keywords(main_kw: str) -> List[str]:
        if images_per_kw <= 1 or lsi_mode == "Off":
            return [main_kw]
        needed = images_per_kw - 1
        extras: List[str] = []
        if lsi_mode == "OpenAI":
            extras = openai_lsi_terms(main_kw, needed, openai_key)
        if not extras and lsi_mode in ("Heuristic","OpenAI"):
            extras = heuristic_lsi(main_kw, needed)
        # Always include main first
        return [main_kw] + extras[:needed]

    # Prepare archive
    zip_buf = io.BytesIO()
    zf = zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED)
    gallery = []

    progress = st.progress(0)
    status = st.empty()

    for i, kw in enumerate(keywords, start=1):
        status.info(f"Working {i}/{len(keywords)}: {kw}")
        all_subkws = expand_keywords(kw)

        for j, active_kw in enumerate(all_subkws, start=1):
            venue = venue_mode_hint(active_kw)
            # Thumbs
            items = collect_thumbnails(active_kw, cand_per_kw)
            with st.expander(f"📷 Thumbnails — choose one or leave 'AI render': {active_kw}", expanded=False):
                picked = show_thumb_picker(active_kw, items)
            if not picked:
                picked = autopick_thumb(items)

            ref_feat = None; expect = {}
            if picked:
                st.caption(f"Using reference: {picked.get('title') or picked['source']} ({picked['source']})")
                feat = analyze_reference(picked["url"])
                if feat.get("ok"):
                    ref_feat = feat; expect = expectation_from_features(feat)

            lock = lock_strength if ref_feat else "Off"
            prompt = build_prompt(SITE_PROFILES.get(site, SITE_PROFILES[DEFAULT_SITE]) and site,
                                  active_kw, season_mode, ref_feat, fidelity, lock, venue)

            tries_per_img = 1 + (fidelity // 40)
            ok_img = None; local_prompt = prompt
            for t in range(tries_per_img):
                try:
                    png = dalle_generate_png_bytes(local_prompt, render_size, openai_key)
                except Exception as e:
                    st.error(f"{active_kw}: OpenAI error — {e}")
                    break
                if ref_feat and lock != "Off":
                    if postcheck(png, expect):
                        ok_img = png; break
                    else:
                        local_prompt += " The required elements must be clearly visible."
                else:
                    ok_img = png; break
            if not ok_img: ok_img = png

            # save blog
            try:
                im = Image.open(io.BytesIO(ok_img)).convert("RGB")
            except Exception:
                st.error(f"{active_kw}: Failed to decode image.")
                continue

            base_slug = slugify(active_kw)
            suffix = "" if len(all_subkws)==1 else f"-{j}"
            fname_blog = f"{base_slug}{suffix}.webp"
            webp_bytes = save_webp_bytes(im, DEFAULT_OUTPUT_W, DEFAULT_OUTPUT_H, webp_quality)
            zf.writestr(fname_blog, webp_bytes)
            gallery.append((fname_blog, webp_bytes))

            # pinterest
            if make_pins:
                fname_pin = f"{base_slug}{suffix}-pin.webp"
                pin_bytes = save_webp_bytes(im, PIN_OUTPUT_W, PIN_OUTPUT_H, webp_quality)
                zf.writestr(fname_pin, pin_bytes)
                gallery.append((fname_pin, pin_bytes))

        progress.progress(i/len(keywords))

    zf.close(); zip_buf.seek(0)
    st.success("Done! Download your images below.")
    st.download_button("⬇️ Download ZIP", data=zip_buf,
                       file_name=f"{slugify(site)}_images.zip", mime="application/zip")

    st.markdown("### Previews & individual downloads")
    cols = st.columns(3)
    for k,(fname, data_bytes) in enumerate(gallery):
        with cols[k % 3]:
            st.image(data_bytes, caption=fname, use_container_width=True)
            st.download_button("Download", data=data_bytes, file_name=fname,
                               mime="image/webp", key=f"dl_{k}")
