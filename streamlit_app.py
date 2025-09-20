# ImageForge Autopilot v3.1.0 — CC Preview
# Bulk blog/Pinterest images with:
# - Hands-free Real/Replica (Creative Commons real photo first via SerpAPI → fallback to AI)
# - NEW: CC Thumbnail Preview & Manual Pick (two-step: Search & Preview → Render)
# - LSI variants (N images per keyword)
# - Season & Corridor Engines (winter/summer & road-trip realism)
# - Pinterest/Blog output presets (WebP)
#
# pip install streamlit requests pillow
# Streamlit secrets (optional): OPENAI_API_KEY, SERPAPI_API_KEY, APP_PASSWORD

import io
import re
import json
import base64
import zipfile
from typing import List, Tuple, Optional, Dict

import requests
from PIL import Image
import streamlit as st

APP_NAME = "ImageForge Autopilot"
APP_VERSION = "v3.1.0 — CC Preview"

st.set_page_config(page_title=f"{APP_NAME} — {APP_VERSION}", layout="wide")

APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")
DEFAULT_SITE = "vailvacay.com"

SITE_PRESETS = {
    "vailvacay.com":  "Colorado alpine resort & village; riverside paths, gondola/base areas, evergreen forests, cozy lodges.",
    "bostonvacay.com":"New England city; Beacon Hill brick, harborwalk, Charles River, landmark stations.",
    "bangkokvacay.com":"Bangkok; temples, canals, night markets, BTS/MRT platforms, bustling street life.",
    "ipetzo.com":      "Pet lifestyle; dogs/cats with owners in parks or cozy homes; neutral interiors.",
    "1-800deals.com":  "Retail/ecommerce; parcels, unboxing, generic products; clean backgrounds.",
}

OUTPUT_PRESETS = {
    "Blog 1200×675 (16:9)": {"w": 1200, "h": 675, "orientation_note": "landscape orientation"},
    "Pinterest 1000×1500 (2:3)": {"w": 1000, "h": 1500, "orientation_note": "portrait/vertical orientation"},
}
DEFAULT_OUTPUT_PRESET = "Blog 1200×675 (16:9)"
DEFAULT_QUALITY = 82
API_IMAGE_SIZE_OPTIONS = ["1536x1024", "1024x1024", "1024x1536"]
SERPAPI_DEFAULT = st.secrets.get("SERPAPI_API_KEY", "")

# ----------------- tiny utils -----------------
def slugify(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[’'`]", "", t)
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t

def crop_to_aspect(img: Image.Image, w: int, h: int) -> Image.Image:
    tr = w / h
    W, H = img.size
    r = W / H
    if r > tr:
        new_w = int(H * tr); left = (W - new_w) // 2
        return img.crop((left, 0, left + new_w, H))
    else:
        new_h = int(W / tr); top = (H - new_h) // 2
        return img.crop((0, top, W, top + new_h))

def to_webp_bytes(img: Image.Image, w: int, h: int, quality: int) -> bytes:
    img = img.convert("RGB")
    img = crop_to_aspect(img, w, h).resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=quality, method=6)
    return buf.getvalue()

def extract_json(txt: str):
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None

# ----------------- OpenAI -----------------
def chat_completion(api_key: str, messages: list, temperature: float = 0.6, model: str = "gpt-4o-mini") -> str:
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=180,
    )
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI chat error {r.status_code}: {r.text}")
    return r.json()["choices"][0]["message"]["content"].strip()

def generate_image_bytes(api_key: str, prompt: str, size: str) -> bytes:
    r = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "gpt-image-1", "prompt": prompt, "size": size},
        timeout=180,
    )
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI image error {r.status_code}: {r.text}")
    d0 = r.json()["data"][0]
    if "b64_json" in d0:
        return base64.b64decode(d0["b64_json"])
    if "url" in d0:
        img = requests.get(d0["url"], timeout=180); img.raise_for_status()
        return img.content
    raise RuntimeError("No image data returned")

# ----------------- SerpAPI (CC images & hints) -----------------
PRICE_TERMS = ["price","prices","cost","ticket","tickets","lift ticket","fee","fees","discount","coupon","how much"]
def keyword_is_pricey(k: str) -> bool:
    kk = k.lower()
    return any(term in kk for term in PRICE_TERMS)

def serpapi_price_hint(api_key: str, query: str) -> Optional[str]:
    try:
        url = "https://serpapi.com/search.json"
        params = {"engine":"google","q":query,"num":5,"api_key":api_key}
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200: return None
        data = r.json()
        texts = []
        ab = data.get("answer_box") or {}
        for k in ("snippet","title","answer"):
            v = ab.get(k)
            if isinstance(v,str): texts.append(v)
        for res in data.get("organic_results",[])[:5]:
            for k in ("snippet","title"):
                v = res.get(k)
                if isinstance(v,str): texts.append(v)
        blob = " ".join(texts)[:1000]
        m = re.findall(r"\$\s*\d{2,4}", blob)
        if m:
            uniq = sorted(set(s.replace(" ","") for s in m))
            return f"Public search hints show figures like: {', '.join(uniq[:4])}. Treat as reference only; do not render numbers."
        return "Public search hints retrieved. Treat as reference only; do not render numbers." if blob else None
    except Exception:
        return None

def serpapi_image_titles(api_key: str, query: str, cc_only: bool = True, max_items: int = 6) -> List[str]:
    try:
        url = "https://serpapi.com/search.json"
        params = {"engine":"google","q":query,"tbm":"isch","ijn":"0","api_key":api_key}
        if cc_only: params["tbs"] = "sur:fc"
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200: return []
        data = r.json()
        results = data.get("images_results",[])[:max_items]
        titles = []
        for it in results:
            t = it.get("title")
            if isinstance(t,str) and t.strip(): titles.append(t.strip())
        return titles
    except Exception:
        return []

def serpapi_cc_image_candidates(api_key: str, query: str, max_items: int = 8) -> List[dict]:
    """Return candidates with original/thumbnail URLs and metadata."""
    try:
        url = "https://serpapi.com/search.json"
        params = {"engine":"google","q":query,"tbm":"isch","ijn":"0","tbs":"sur:fc","api_key":api_key}
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200: return []
        data = r.json()
        out = []
        for it in data.get("images_results",[])[:max_items]:
            original = it.get("original") or it.get("thumbnail") or it.get("source")
            thumb   = it.get("thumbnail") or it.get("original") or it.get("source")
            if not original: continue
            width  = it.get("original_width") or it.get("width") or 0
            height = it.get("original_height") or it.get("height") or 0
            out.append({
                "original_url": original,
                "thumbnail_url": thumb,
                "width": int(width or 0),
                "height": int(height or 0),
                "title": it.get("title") or "",
                "source": it.get("source") or "",
            })
        return out
    except Exception:
        return []

def choose_best_cc_image(cands: List[dict]) -> Optional[dict]:
    if not cands: return None
    return sorted(cands, key=lambda c: (c.get("width",0)*c.get("height",0)), reverse=True)[0]

def safe_download_image(url: str) -> Optional[Image.Image]:
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200: return None
        return Image.open(io.BytesIO(resp.content))
    except Exception:
        return None

# ----------------- Season Engine -----------------
MONTH_ALIASES = {
    "january":"jan","february":"feb","march":"mar","april":"apr","may":"may","june":"jun",
    "july":"jul","august":"aug","september":"sep","october":"oct","november":"nov","december":"dec",
    "jan":"jan","feb":"feb","mar":"mar","apr":"apr","jun":"jun","jul":"jul","aug":"aug","sep":"sep","oct":"oct","nov":"nov","dec":"dec"
}
def extract_month_token(text: str) -> Optional[str]:
    t = text.lower()
    for w in re.findall(r"[a-z]+", t):
        if w in MONTH_ALIASES: return MONTH_ALIASES[w]
    return None

def skiish(keyword: str) -> bool:
    k = keyword.lower()
    return any(s in k for s in ["ski","back bowl","lift ticket","gondola","powder","snowboard","chairlift"])

def season_hint_for_site(site_key: str, keyword: str) -> Optional[str]:
    m = extract_month_token(keyword); k = keyword.lower()
    if site_key=="vailvacay.com":
        if skiish(k): return "Season: winter skiing. Snowy runs and operating lifts; winter clothing; avoid summer flowers/green meadows."
        if m in {"nov","dec","jan","feb","mar","apr"}:
            if m=="apr": return "Season: late-season spring skiing—snowy runs still active; sunny OK but no summer flowers."
            return "Season: winter—snow on ski slopes; winter outfits."
        if m=="may": return "Season: shoulder—town thawed, high peaks/ski runs may retain snow; avoid full-summer flowers."
        if m in {"jun","jul","aug"}: return "Season: summer—no town snow; peaks minimal patches."
        if m=="sep": return "Season: early fall—greens fading; minimal/no snow except high peaks."
        if m=="oct": return "Season: fall—aspens turning; possible light peak dusting; no active skiing unless requested."
        if "winter" in k: return "Season: winter—snowy slopes."
        if "summer" in k: return "Season: summer—no town snow."
        if "spring" in k: return "Season: mountain spring—snow can persist on runs/peaks."
    if site_key=="bostonvacay.com" and m:
        if m in {"dec","jan","feb"}: return "Season: Boston winter; snow optional."
        if m in {"jun","jul","aug"}: return "Season: summer by harbor/Charles."
        if m in {"sep","oct"}: return "Season: fall foliage tones."
        if m in {"apr","may"}: return "Season: spring blooms."
    if site_key=="bangkokvacay.com": return "Season: tropical warm; no coats or winter scenes."
    return None

# ----------------- Corridor Engine -----------------
ROUTE_SW_KEYWORDS = ["albuquerque","santa fe","taos","new mexico","nm","rio grande","gorge","mesa","pueblo","adobe","sand dunes","great sand dunes"]
def parse_route_endpoints(keyword: str) -> Optional[Tuple[str,str]]:
    k = " ".join(keyword.lower().split())
    m = re.search(r"between\s+(.+?)\s+and\s+(.+)$", k)
    if m: return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"from\s+(.+?)\s+to\s+(.+)$", k)
    if m: return m.group(1).strip(), m.group(2).strip()
    return None

def corridor_hint_for_route(site_key: str, keyword: str) -> Optional[str]:
    endpoints = parse_route_endpoints(keyword)
    if not endpoints: return None
    k = keyword.lower()
    generic = ("Road-trip corridor scene: highway or scenic overlook, or dash+map angle. "
               "Brown park sign/mile marker in soft focus. No brand signage; avoid resort artifacts like gondolas/chairlifts/base lodges.")
    if any(sw in k for sw in ROUTE_SW_KEYWORDS) or ("vail" in k and "albuquerque" in k):
        return (generic + " Southwest palette: sage/piñon-juniper, red/orange mesas, adobe/Spanish hints; "
                "icons like Rio Grande Gorge Bridge or Great Sand Dunes distant. No alpine gondolas.")
    return generic

# ----------------- Planner + Critic -----------------
PLANNER_BASE = """
You are a senior creative director writing photorealistic image briefs for a travel/consumer blog.
Given a keyword and a site vibe, craft ONE concise prompt (1–2 sentences) for a DALLE-like model.
Use editorial stock-photo style; balanced composition; natural light; {ORIENTATION_NOTE}.
No readable logos or typography. Privacy-respecting. Family-safe.

Nudges/examples:
- "what county is …": tabletop regional map with a pin near the destination OR county courthouse exterior with seal out of focus (no readable text).
- "what river runs through …": make the river the hero; town context secondary.
- "what mountain range is … in": emphasize ridgelines/peaks; town minimal.
- "how far/between/from …": road-trip planning vibe (highway/overlook; optional dash+map); keep brand signage unreadable.
"""
SITE_NUDGES = {
    "vailvacay.com": [
        "If topic implies skiing (back bowls/lift tickets), show snowy slopes and lift infrastructure.",
        "For ticket/price topics: conceptual scene (blurred board/phone); NEVER readable numbers."
    ],
    "bostonvacay.com": [
        "Ferry/Bar Harbor: terminal/ramp scene with vessel; signage generic/debranded.",
    ],
    "bangkokvacay.com": [
        "BTS/Chinatown: platform train arrival; wayfinding shapes suggestive but unreadable.",
    ],
    "ipetzo.com": ["Dog-friendly: patio/lobby seating; water bowl; no hotel branding."],
    "1-800deals.com": ["Deals/unboxing: generic parcels/product tableaus; device screens out-of-focus; no trademarks."],
}
CRITIC_SYSTEM = """
You are a prompt critic. Ensure the prompt clearly illustrates the keyword, honors any season/corridor directives,
keeps brand-neutrality (no readable logos/text), and is concise. For price/ticket topics, forbid readable numbers.
If it needs improvement, return {"action":"refine","prompt":"..."}; otherwise {"action":"ok"}.
"""

def build_planner_system(site_key: str,
                         facts_hint: Optional[str],
                         ref_cues: List[str],
                         season_hint: Optional[str],
                         corridor_hint: Optional[str],
                         orientation_note: str) -> str:
    nudges = SITE_NUDGES.get(site_key, [])
    site_block = "\nSite-specific nudges:\n- " + "\n- ".join(nudges) + "\n" if nudges else ""
    facts_block = f"\nContext (for concept only; do NOT render text/numbers):\n- {facts_hint}\n" if facts_hint else ""
    season_block = f"\nSeason directive (MUST follow):\n- {season_hint}\n" if season_hint else ""
    corridor_block = f"\nCorridor directive (MUST follow):\n- {corridor_hint}\n" if corridor_hint else ""
    cues_block = ("\nReference cues (from public image titles; inspiration only; no text/logos):\n- "
                  + "\n- ".join(ref_cues[:6]) + "\n") if ref_cues else ""
    return (PLANNER_BASE.replace("{ORIENTATION_NOTE}", orientation_note)
            + site_block + season_block + corridor_block + facts_block + cues_block
            + '\nOutput JSON ONLY:\n{"prompt": "<final one- or two-sentence prompt>"}')

def plan_prompt(api_key: str, site_vibe: str, keyword: str, site_key: str,
                facts_hint: Optional[str], ref_cues: List[str],
                season_hint: Optional[str], corridor_hint: Optional[str],
                orientation_note: str) -> str:
    sys = build_planner_system(site_key, facts_hint, ref_cues, season_hint, corridor_hint, orientation_note)
    user = f"Site vibe: {site_vibe}\nKeyword: {keyword}\nWrite one concise photorealistic prompt."
    content = chat_completion(api_key, [{"role":"system","content":sys},{"role":"user","content":user}], temperature=0.4)
    data = extract_json(content)
    return (data or {}).get("prompt", content)

def critique_and_refine(api_key: str, keyword: str, prompt: str) -> str:
    user = f"Keyword: {keyword}\nProposed prompt: {prompt}"
    content = chat_completion(api_key, [{"role":"system","content":CRITIC_SYSTEM},{"role":"user","content":user}], temperature=0.2)
    data = extract_json(content)
    if data and data.get("action")=="refine" and data.get("prompt"):
        return data["prompt"]
    return prompt

# ----------------- LSI variants -----------------
def generate_lsi_variants(api_key: str, main_keyword: str, site_key: str, count: int) -> List[str]:
    if count <= 0: return []
    system = ("You are an SEO strategist. Given a main keyword, produce concise LSI variants suitable for distinct images. "
              "Keep any month/season/location in place. Avoid brands/names/explicit content. "
              "Return JSON ONLY: {\"variants\": [\"...\"]}")
    site_nudge = {
        "vailvacay.com": "Prefer scenery/food/family/kid-friendly/gear angles.",
        "bostonvacay.com": "Prefer city/harbor/food/transport angles.",
        "bangkokvacay.com": "Prefer transit/markets/temples/food (family-safe).",
        "ipetzo.com": "Prefer pet-friendly/outdoors/indoor play/grooming/gear.",
        "1-800deals.com": "Prefer generic shopping/unboxing/price-checking scenes (no brands).",
    }.get(site_key, "")
    user = f"Main keyword: {main_keyword}\nSite: {site_key}\nCount: {count}\nGuidance: {site_nudge}"
    resp = chat_completion(api_key, [{"role":"system","content":system},{"role":"user","content":user}], temperature=0.5)
    data = extract_json(resp) or {}
    raw = data.get("variants", []) if isinstance(data.get("variants", []), list) else []
    out, seen = [], {main_keyword.strip().lower()}
    for v in raw:
        if not isinstance(v,str): continue
        s = v.strip()
        if not s or s.lower() in seen: continue
        seen.add(s.lower()); out.append(s[:140])
        if len(out)>=count: break
    return out

# ----------------- Place detection -----------------
VENUE_HINTS = [
    "restaurant","tavern","bar","cafe","coffee","roaster","bakery","diner","eatery","steakhouse","sushi",
    "hotel","lodge","inn","resort","hostel","motel","spa",
    "shop","store","boutique","outfitter","market","mall","plaza","center","centre",
    "museum","gallery","theater","theatre","park","trail","gondola","terminal","station","ferry","pier",
]
def looks_like_place_query(keyword: str) -> bool:
    k = keyword.lower()
    if any(h in k for h in VENUE_HINTS): return True
    if "," in k and any(city in k for city in ["vail","boston","bangkok","colorado","massachusetts","thailand"]): return True
    words = re.findall(r"[A-Za-z][A-Za-z']+", keyword)
    caps = sum(1 for w in words if w[:1].isupper())
    return caps >= 3

def build_cc_queries(keyword: str, site_key: str) -> List[str]:
    base_city = {
        "vailvacay.com": "Vail Colorado",
        "bostonvacay.com": "Boston",
        "bangkokvacay.com": "Bangkok",
        "ipetzo.com": "",
        "1-800deals.com": ""
    }.get(site_key, "")
    q0 = f"{keyword} {base_city}".strip()
    variants = [q0,
                f"{keyword} storefront {base_city}".strip(),
                f"{keyword} exterior {base_city}".strip(),
                f"{keyword} sign {base_city}".strip(),
                f"\"{keyword}\" {base_city}".strip()]
    seen, out = set(), []
    for q in variants:
        if q and q not in seen:
            out.append(q); seen.add(q)
    return out

def choose_render_size_for_orientation(chosen_size: str, orientation_note: str) -> str:
    portrait = "portrait" in orientation_note
    if portrait and chosen_size != "1024x1536": return "1024x1536"
    if (not portrait) and chosen_size == "1024x1536": return "1536x1024"
    return chosen_size

# ----------------- Streamlit UI -----------------
st.title(f"{APP_NAME} — {APP_VERSION}")
st.caption("Enable **Preview CC candidates** to see thumbnails and manually pick which real photo to use. "
           "If none is picked (or none found), we generate a brand-neutral AI replica. "
           "Season & Corridor Engines included. LSI supported.")

if APP_PASSWORD:
    if st.text_input("Team password", type="password") != APP_PASSWORD:
        st.stop()

with st.sidebar:
    site = st.selectbox("Site", list(SITE_PRESETS.keys()),
                        index=list(SITE_PRESETS.keys()).index(DEFAULT_SITE))
    output_preset = st.selectbox("Output preset", list(OUTPUT_PRESETS.keys()),
                                 index=list(OUTPUT_PRESETS.keys()).index(DEFAULT_OUTPUT_PRESET))
    images_per_kw = st.number_input("Images per keyword", min_value=1, max_value=12, value=1, step=1)
    quality = st.slider("WebP quality", 60, 95, DEFAULT_QUALITY)

    render_size = st.selectbox("Model render size (internal)", API_IMAGE_SIZE_OPTIONS, index=0)

    source_mode = st.radio("Source mode",
                           ["Hands-free Real/Replica (CC first)", "AI only", "Real-only CC (skip if none)"])
    preview_cc = st.checkbox("Preview CC candidates (manual pick)", value=True,
                             help="Shows 3–8 CC thumbnails per place keyword. You choose which one to use "
                                  "or select AI replica/Skip per item.")

    serpapi_key = st.text_input("SERPAPI_API_KEY (optional)", type="password",
                                value=st.secrets.get("SERPAPI_API_KEY",""))
openai_key = st.text_input("OpenAI API key", type="password",
                           value=st.secrets.get("OPENAI_API_KEY",""))

keywords_text = st.text_area("Keywords (one per line)", height=240,
                             placeholder="Tavern on the Square, Vail Colorado\nBest seafood restaurant in Boston")

c1, c2, c3 = st.columns([1,1,1])
btn_search = c1.button("Search & Preview" if preview_cc else "Generate")
btn_render = c2.button("Render Images")
btn_clear = c3.button("Clear")

if btn_clear:
    st.session_state.clear()
    st.experimental_rerun()

# ----------------- Session state buckets -----------------
ss = st.session_state
ss.setdefault("work_items", [])             # list of (main_kw, actual_kw, work_id)
ss.setdefault("lsi_record", {})             # main_kw -> [lsi]
ss.setdefault("cc_candidates", {})          # work_id -> [candidate dict]
ss.setdefault("cc_pick", {})                # work_id -> {"mode": "cc"/"ai"/"skip", "index": int}
ss.setdefault("out_cfg", OUTPUT_PRESETS[output_preset])
ss.setdefault("render_plan_ready", False)

# ----------------- AI generation helpers -----------------
def do_ai_image(api_key: str, keyword: str, site_key: str,
                out_cfg: dict, chosen_render_size: str,
                serpapi_key: str) -> Tuple[str, bytes, Optional[str], List[str]]:
    site_vibe = SITE_PRESETS.get(site_key, SITE_PRESETS[DEFAULT_SITE])
    W, H = out_cfg["w"], out_cfg["h"]
    orientation_note = out_cfg["orientation_note"]
    render_sz = choose_render_size_for_orientation(chosen_render_size, orientation_note)

    season_hint = season_hint_for_site(site_key, keyword)
    corridor_hint = corridor_hint_for_route(site_key, keyword)

    facts_hint = None
    if serpapi_key and keyword_is_pricey(keyword):
        facts_hint = serpapi_price_hint(serpapi_key, f"{keyword} {site_key.split('.')[0]}")

    ref_titles = serpapi_image_titles(serpapi_key, f"{keyword} {site_key.split('.')[0]}", cc_only=True, max_items=5) if serpapi_key else []
    ref_cues = []
    if ref_titles:
        sys = ("You are a visual summarizer. Given image titles, produce 3–6 neutral cues "
               "that help illustrate the topic without logos/text. JSON only: {\"cues\":[\"...\"]}")
        content = chat_completion(api_key,
                                  [{"role":"system","content":sys},
                                   {"role":"user","content":"Keyword: "+keyword+"\nTitles:\n- "+"\n- ".join(ref_titles)}],
                                  temperature=0.2)
        data = extract_json(content) or {}
        ref_cues = [str(c)[:100] for c in data.get("cues",[]) if isinstance(c,str)][:6]

    base = plan_prompt(api_key, site_vibe, keyword, site_key, facts_hint, ref_cues, season_hint, corridor_hint, orientation_note)
    prompt = critique_and_refine(api_key, keyword, base)

    png = generate_image_bytes(api_key, prompt, render_sz)
    img = Image.open(io.BytesIO(png))
    return prompt, to_webp_bytes(img, W, H, quality), facts_hint, ref_cues

# ----------------- Step 1: Search & Preview -----------------
if btn_search:
    if preview_cc and source_mode == "AI only":
        st.info("Preview is off for AI-only mode. Switch to a Real/Replica mode to use CC preview.")

    # Build work items + LSI
    out_cfg = OUTPUT_PRESETS[output_preset]
    ss["out_cfg"] = out_cfg
    base_kws = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
    if not base_kws:
        st.warning("Please paste at least one keyword.")
        st.stop()

    work_items = []
    lsi_record = {}
    for main_kw in base_kws:
        need = max(0, images_per_kw - 1)
        lsi = generate_lsi_variants(openai_key, main_kw, site, need) if (need>0 and openai_key) else []
        lsi_record[main_kw] = lsi
        work_items.append((main_kw, main_kw, slugify(main_kw)))
        for v in lsi:
            work_items.append((main_kw, v, f"{slugify(main_kw)}--{slugify(v)}"))

    ss["work_items"] = work_items
    ss["lsi_record"] = lsi_record
    ss["cc_candidates"] = {}
    ss["cc_pick"] = {}
    ss["render_plan_ready"] = False

    # If preview enabled and not AI-only, fetch candidates
    if preview_cc and source_mode != "AI only" and serpapi_key:
        st.success("CC search started…")
        for _main, actual, work_id in work_items:
            if looks_like_place_query(actual):
                queries = build_cc_queries(actual, site)
                all_cands = []
                for q in queries:
                    cands = serpapi_cc_image_candidates(serpapi_key, q, max_items=8)
                    # prefer unique (by original_url)
                    seen, uniq = set(), []
                    for c in cands:
                        url = c.get("original_url")
                        if url and url not in seen:
                            uniq.append(c); seen.add(url)
                    all_cands.extend(uniq)
                    if len(all_cands) >= 8:
                        break
                ss["cc_candidates"][work_id] = all_cands[:8]
            else:
                ss["cc_candidates"][work_id] = []  # not a place or no candidates
        st.toast("CC candidates fetched. Review below and set your choices, then click **Render Images**.", icon="✅")
    else:
        # No preview path → go straight to render on next click
        st.info("No CC preview (either disabled, AI-only, or missing SerpAPI). Click **Render Images** to generate.")

# ----------------- CC Preview UI -----------------
if preview_cc and source_mode != "AI only" and ss["work_items"]:
    st.markdown("## CC Candidates (manual pick)")
    for (main_kw, actual_kw, work_id) in ss["work_items"]:
        cands = ss["cc_candidates"].get(work_id, [])
        if not cands:
            continue  # nothing to preview for this item
        st.markdown(f"**{actual_kw}**")
        # radio: choose AI/Skip or one of candidates by index
        opt_labels = []
        opt_values = []
        for i, c in enumerate(cands):
            title = (c.get("title") or "").strip()
            host  = (c.get("source") or "").strip()
            label = f"CC #{i+1} — {title[:70]} [{' from ' + host if host else ''}]"
            opt_labels.append(label); opt_values.append(f"cc:{i}")
        # extra options depending on source mode
        if source_mode == "Real-only CC (skip if none)":
            opt_labels = ["Skip this item"] + opt_labels
            opt_values = ["skip"] + opt_values
            default = "skip"
        else:
            opt_labels = ["AI replica"] + opt_labels
            opt_values = ["ai"] + opt_values
            default = "ai"

        choice = st.radio("Pick one:", options=opt_values, index=0,
                          format_func=lambda v: opt_labels[opt_values.index(v)],
                          key=f"pick_{work_id}")
        ss["cc_pick"][work_id] = {"mode": choice.split(":")[0],
                                  "index": int(choice.split(":")[1]) if choice.startswith("cc:") else -1}

        # show thumbs in a row
        cols = st.columns(4)
        for i, c in enumerate(cands[:8]):
            with cols[i % 4]:
                st.image(c.get("thumbnail_url") or c.get("original_url"),
                         caption=f"CC #{i+1}", use_container_width=True)

    if ss["cc_candidates"]:
        ss["render_plan_ready"] = True

# ----------------- Step 2: Render -----------------
if btn_render:
    if not ss["work_items"]:
        st.warning("Click **Search & Preview** first (or disable preview and click **Generate**).")
        st.stop()

    out_cfg = ss["out_cfg"]
    W, H = out_cfg["w"], out_cfg["h"]

    # If preview was requested and we have candidates, require picks (or defaulted earlier)
    if preview_cc and source_mode != "AI only" and not ss.get("render_plan_ready"):
        st.warning("No CC candidates were prepared yet. Click **Search & Preview** first.")
        st.stop()

    # Render loop
    prog = st.progress(0); done = 0
    thumbs: List[Tuple[str, bytes]] = []
    meta_log: List[str] = []
    prompts_used: List[Tuple[str, str]] = []
    facts_notes: List[Tuple[str, str]] = []
    ref_notes: List[Tuple[str, List[str]]] = []
    attribution_lines: List[str] = []

    zip_buf = io.BytesIO()
    zipf = zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED)

    for (main_kw, actual_kw, work_id) in ss["work_items"]:
        fname_base = work_id
        used_real = False
        webp_bytes = None
        prompt_used = None

        tried_cc = False
        cands = ss["cc_candidates"].get(work_id, [])
        pick = ss["cc_pick"].get(work_id, {"mode":"ai","index":-1})

        # Non-preview modes: emulate old behavior
        if not preview_cc or source_mode=="AI only" or not cands:
            if source_mode != "AI only" and serpapi_key and looks_like_place_query(actual_kw):
                tried_cc = True
                # hands-free or real-only auto pick
                if cands:
                    best = choose_best_cc_image(cands)
                else:
                    # fetch ad-hoc (no preview)
                    allq = build_cc_queries(actual_kw, site)
                    allc = []
                    for q in allq:
                        allc.extend(serpapi_cc_image_candidates(serpapi_key, q, max_items=6))
                    best = choose_best_cc_image(allc)
                if best:
                    img = safe_download_image(best["original_url"])
                    if img:
                        webp_bytes = to_webp_bytes(img, W, H, quality)
                        used_real = True
                        meta_log.append(f"{actual_kw} → CC photo (auto)")
                        attribution_lines.append(
                            f"{fname_base} -- CC via Google Images (SerpAPI): {best.get('original_url','')} | "
                            f"Title: {best.get('title','')} | Source: {best.get('source','')}"
                        )

            if not used_real:
                if source_mode == "Real-only CC (skip if none)" and tried_cc:
                    meta_log.append(f"{actual_kw} → skipped (no CC photo)")
                    done += 1; prog.progress(done/len(ss['work_items'])); continue
                # AI
                if not openai_key:
                    st.error("OpenAI key required for AI replica.")
                    st.stop()
                try:
                    prompt_used, webp_bytes, facts_hint, ref_cues = do_ai_image(
                        openai_key, actual_kw, site, out_cfg, render_size, serpapi_key
                    )
                    prompts_used.append((f"{fname_base}.webp", prompt_used))
                    if facts_hint: facts_notes.append((fname_base, facts_hint))
                    if ref_cues: ref_notes.append((fname_base, ref_cues))
                    meta_log.append(f"{actual_kw} → AI replica")
                except Exception as e:
                    st.error(f"{actual_kw}: {e}")
                    done += 1; prog.progress(done/len(ss['work_items'])); continue

        else:
            # Preview mode with candidates & a pick
            mode = pick.get("mode","ai")
            idx  = pick.get("index",-1)
            if mode == "cc" and 0 <= idx < len(cands):
                tried_cc = True
                cand = cands[idx]
                img = safe_download_image(cand["original_url"])
                if img:
                    webp_bytes = to_webp_bytes(img, W, H, quality)
                    used_real = True
                    meta_log.append(f"{actual_kw} → CC photo (manual pick)")
                    attribution_lines.append(
                        f"{fname_base} -- CC via Google Images (SerpAPI): {cand.get('original_url','')} | "
                        f"Title: {cand.get('title','')} | Source: {cand.get('source','')}"
                    )
                else:
                    meta_log.append(f"{actual_kw} → chosen CC failed to download; using AI replica")
            elif mode == "skip":
                meta_log.append(f"{actual_kw} → skipped by user")
                done += 1; prog.progress(done/len(ss['work_items'])); continue

            if not used_real:
                if source_mode == "Real-only CC (skip if none)":
                    meta_log.append(f"{actual_kw} → skipped (no CC used)")
                    done += 1; prog.progress(done/len(ss['work_items'])); continue
                if not openai_key:
                    st.error("OpenAI key required for AI replica."); st.stop()
                try:
                    prompt_used, webp_bytes, facts_hint, ref_cues = do_ai_image(
                        openai_key, actual_kw, site, out_cfg, render_size, serpapi_key
                    )
                    prompts_used.append((f"{fname_base}.webp", prompt_used))
                    if facts_hint: facts_notes.append((fname_base, facts_hint))
                    if ref_cues: ref_notes.append((fname_base, ref_cues))
                    meta_log.append(f"{actual_kw} → AI replica")
                except Exception as e:
                    st.error(f"{actual_kw}: {e}")
                    done += 1; prog.progress(done/len(ss['work_items'])); continue

        # write file
        if webp_bytes:
            suffix = "photo-cc" if used_real else "replica"
            out_name = f"{fname_base}--{suffix}.webp" if tried_cc or preview_cc else f"{fname_base}.webp"
            zipf.writestr(out_name, webp_bytes)
            thumbs.append((out_name, webp_bytes))

        done += 1
        prog.progress(done/len(ss["work_items"]))

    # finalize ZIP
    if attribution_lines:
        zipf.writestr("ATTRIBUTION.txt", "\n".join(attribution_lines))
    zipf.close(); zip_buf.seek(0)

    if not thumbs:
        st.error("No images were produced.")
    else:
        st.success("Done! Download your images below.")
        st.download_button("⬇️ Download ZIP", data=zip_buf,
                           file_name=f"{slugify(site)}_{slugify(output_preset)}.zip",
                           mime="application/zip")

        st.markdown("### Previews")
        cols = st.columns(3 if H <= W else 2)
        for i,(name, data_bytes) in enumerate(thumbs):
            with cols[i % len(cols)]:
                st.image(data_bytes, caption=name, use_container_width=True)
                st.download_button("Download", data=data_bytes, file_name=name, mime="image/webp", key=f"dl_{i}")

        if meta_log:
            with st.expander("Run log"):
                for line in meta_log: st.write("• " + line)

        if ss["lsi_record"]:
            with st.expander("LSI variants generated"):
                for main_kw, variants in ss["lsi_record"].items():
                    st.markdown(f"**{main_kw}**")
                    st.write(", ".join(variants) if variants else "_None_")

        if prompts_used:
            with st.expander("Prompts used (AI only)"):
                for fname, p in prompts_used:
                    st.markdown(f"**{fname}**"); st.code(p, language="text")

        if facts_notes:
            with st.expander("Price facts assist (reference only)"):
                for fname, note in facts_notes:
                    st.markdown(f"**{fname}**"); st.write(note)

        if ref_notes:
            with st.expander("Reference cues used"):
                for fname, cues in ref_notes:
                    st.markdown(f"**{fname}**"); st.write(", ".join(cues))
