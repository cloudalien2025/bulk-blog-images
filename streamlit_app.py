# ImageForge Autopilot v3.4.0 — Maps Resolver + Openverse
# Bulk blog/Pinterest images with:
# • Real CC photos from Google Images (SerpAPI), Flickr (text & geo), and Openverse (free)
# • Place Resolver (SerpAPI Google Maps) -> better venue synonyms + lat/lon for geo search
# • Thumbnail Preview & Manual Pick (or hands-free)
# • AI replica fallback with Season & Corridor engines
# • LSI variants per keyword, Pinterest/Blog presets, price-topic guardrails
#
# pip install streamlit requests pillow
# Optional Streamlit secrets: OPENAI_API_KEY, SERPAPI_API_KEY, FLICKR_API_KEY, APP_PASSWORD

import io, re, json, base64, zipfile
from typing import List, Tuple, Optional, Dict
import requests
from PIL import Image
import streamlit as st

APP_NAME    = "ImageForge Autopilot"
APP_VERSION = "v3.4.0 — Maps Resolver + Openverse"

st.set_page_config(page_title=f"{APP_NAME} — {APP_VERSION}", layout="wide")

# ---------- Config ----------
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
FLICKR_DEFAULT  = st.secrets.get("FLICKR_API_KEY", "")

# ---------- Utils ----------
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

def safe_download_image(url: str) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=25)
        if r.status_code != 200: return None
        return Image.open(io.BytesIO(r.content))
    except Exception:
        return None

# ---------- OpenAI ----------
def chat_completion(api_key: str, messages: list, temperature: float = 0.6, model: str = "gpt-4o-mini") -> str:
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature},
        timeout=180,
    )
    if r.status_code != 200: raise RuntimeError(f"OpenAI chat error {r.status_code}: {r.text}")
    return r.json()["choices"][0]["message"]["content"].strip()

def generate_image_bytes(api_key: str, prompt: str, size: str) -> bytes:
    r = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "gpt-image-1", "prompt": prompt, "size": size},
        timeout=180,
    )
    if r.status_code != 200: raise RuntimeError(f"OpenAI image error {r.status_code}: {r.text}")
    d0 = r.json()["data"][0]
    if "b64_json" in d0: return base64.b64decode(d0["b64_json"])
    if "url" in d0:
        img = requests.get(d0["url"], timeout=180); img.raise_for_status()
        return img.content
    raise RuntimeError("No image data returned")

# ---------- SerpAPI: Google Images (CC) ----------
def serpapi_cc_images(api_key: str, query: str, max_items: int = 8) -> List[dict]:
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
            out.append({
                "source": "GoogleCC",
                "original_url": original,
                "thumbnail_url": thumb,
                "title": it.get("title") or "",
                "host": it.get("source") or "",
                "license": "Creative Commons (as reported by Google Images)",
                "attribution_url": original,
                "owner_name": "",
            })
        return out
    except Exception:
        return []

# ---------- SerpAPI: Maps Place Resolver ----------
def serpapi_place_resolve(api_key: str, query: str) -> Optional[dict]:
    """
    Returns: {"name": str, "lat": float, "lng": float, "alt_names": [..]}
    """
    try:
        url = "https://serpapi.com/search.json"
        params = {"engine":"google_maps","q":query,"type":"search","api_key":api_key}
        r = requests.get(url, params=params, timeout=25)
        if r.status_code != 200: return None
        data = r.json()
        alt = []
        place = data.get("place_results") or {}
        if not place:
            # fallback to first local_result
            locs = data.get("local_results") or []
            place = locs[0] if locs else {}
        if not place: return None
        gps  = place.get("gps_coordinates") or {}
        name = place.get("title") or place.get("name") or query
        kg   = data.get("knowledge_graph") or {}
        for k in ("title","subtitle","type","parent_company"):
            v = kg.get(k)
            if isinstance(v,str): alt.append(v)
        addr = place.get("address")
        if isinstance(addr,str): alt.append(addr)
        # de-dup and clean
        seen, alts = set(), []
        for a in alt:
            a = a.strip()
            if a and a not in seen:
                alts.append(a); seen.add(a)
        return {
            "name": name,
            "lat": gps.get("latitude"),
            "lng": gps.get("longitude"),
            "alt_names": alts
        }
    except Exception:
        return None

# ---------- Flickr (CC: text + geo) ----------
FLICKR_LICENSE_IDS = "4,5,7,8,9,10"
FLICKR_LICENSE_MAP = {
    "4": ("CC BY 2.0",  "https://creativecommons.org/licenses/by/2.0/"),
    "5": ("CC BY-SA 2.0","https://creativecommons.org/licenses/by-sa/2.0/"),
    "7": ("No known copyright restrictions", "https://www.flickr.com/commons/usage/"),
    "8": ("United States Government Work", "https://www.usa.gov/government-works"),
    "9": ("CC0 1.0", "https://creativecommons.org/publicdomain/zero/1.0/"),
    "10": ("Public Domain Mark 1.0", "https://creativecommons.org/publicdomain/mark/1.0/"),
}

def flickr_cc_text(api_key: str, query: str, max_items: int = 10) -> List[dict]:
    if not api_key: return []
    try:
        url = "https://www.flickr.com/services/rest/"
        params = {
            "method": "flickr.photos.search",
            "api_key": api_key,
            "text": query,
            "sort": "relevance",
            "per_page": max_items,
            "license": FLICKR_LICENSE_IDS,
            "content_type": 1,
            "media": "photos",
            "safe_search": 1,
            "extras": "url_o,url_l,url_c,url_m,owner_name,license,path_alias",
            "format": "json", "nojsoncallback": 1,
        }
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200: return []
        return _flickr_parse(r.json())
    except Exception:
        return []

def flickr_cc_geo(api_key: str, lat: float, lng: float, tags: str, radius_km: float = 0.6, max_items: int = 12) -> List[dict]:
    if not api_key: return []
    try:
        url = "https://www.flickr.com/services/rest/"
        params = {
            "method": "flickr.photos.search",
            "api_key": api_key,
            "lat": lat, "lon": lng,
            "radius": radius_km, "radius_units": "km",
            "tags": tags, "tag_mode": "any",
            "sort": "relevance",
            "per_page": max_items,
            "license": FLICKR_LICENSE_IDS,
            "content_type": 1,
            "media": "photos",
            "safe_search": 1,
            "extras": "url_o,url_l,url_c,url_m,owner_name,license,path_alias",
            "format": "json", "nojsoncallback": 1,
        }
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200: return []
        return _flickr_parse(r.json())
    except Exception:
        return []

def _flickr_parse(data: dict) -> List[dict]:
    photos = (data.get("photos") or {}).get("photo", [])
    out = []
    for p in photos:
        original = p.get("url_o") or p.get("url_l") or p.get("url_c") or p.get("url_m")
        thumb    = p.get("url_m") or p.get("url_c") or p.get("url_l") or p.get("url_o")
        if not original: continue
        pid = p.get("id")
        alias = p.get("path_alias") or p.get("owner") or ""
        page = f"https://www.flickr.com/photos/{alias}/{pid}" if pid else "https://www.flickr.com/"
        lic_id = str(p.get("license",""))
        lic_name, lic_url = FLICKR_LICENSE_MAP.get(lic_id, ("Creative Commons", "https://creativecommons.org/"))
        out.append({
            "source":"Flickr","original_url":original,"thumbnail_url":thumb,
            "title": p.get("title") or "","host":"Flickr","owner_name":p.get("ownername") or "",
            "license": lic_name, "license_url": lic_url, "attribution_url": page
        })
    return out

# ---------- Openverse (free CC) ----------
def openverse_cc_images(query: str, max_items: int = 10) -> List[dict]:
    try:
        url = "https://api.openverse.engineering/v1/images/"
        params = {"q": query, "license_type": "commercial", "page_size": max_items}
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200: return []
        res = r.json().get("results", [])
        out = []
        for it in res[:max_items]:
            original = it.get("url") or it.get("thumbnail")
            thumb    = it.get("thumbnail") or it.get("url")
            if not original: continue
            out.append({
                "source":"Openverse",
                "original_url": original,
                "thumbnail_url": thumb,
                "title": it.get("title") or "",
                "host": it.get("source") or "",
                "owner_name": it.get("creator") or "",
                "license": (it.get("license") or "").upper(),
                "license_url": it.get("license_url") or "",
                "attribution_url": it.get("foreign_landing_url") or original,
            })
        return out
    except Exception:
        return []

# ---------- Price hint ----------
PRICE_TERMS = ["price","prices","cost","ticket","tickets","lift ticket","fee","fees","discount","coupon","how much"]
def is_price_topic(k: str) -> bool:
    k = k.lower(); return any(t in k for t in PRICE_TERMS)

def serpapi_price_hint(api_key: str, query: str) -> Optional[str]:
    try:
        url = "https://serpapi.com/search.json"
        params = {"engine":"google","q":query,"num":5,"api_key":api_key}
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200: return None
        data = r.json(); texts = []
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

# ---------- Season & Corridor Engines ----------
MONTH_ALIASES = {m:m[:3] for m in
    ["january","february","march","april","may","june","july","august","september","october","november","december"]}
for k,v in list(MONTH_ALIASES.items()):
    MONTH_ALIASES[v]=v

def extract_month_token(t: str) -> Optional[str]:
    t = t.lower()
    for w in re.findall(r"[a-z]+", t):
        if w in MONTH_ALIASES: return MONTH_ALIASES[w]
    return None

def skiish(k: str) -> bool:
    k = k.lower()
    return any(s in k for s in ["ski","back bowl","lift ticket","gondola","powder","snowboard","chairlift"])

def season_hint_for_site(site: str, keyword: str) -> Optional[str]:
    m = extract_month_token(keyword); k = keyword.lower()
    if site=="vailvacay.com":
        if skiish(k): return "Season: winter skiing. Snowy runs and operating lifts; winter clothing; avoid summer flowers/green meadows."
        if m in {"nov","dec","jan","feb","mar","apr"}:
            return "Season: late-winter/spring skiing OK; show snow on runs; avoid summer scenery." if m=="apr" else \
                   "Season: winter—snow on slopes; winter outfits."
        if m=="may": return "Season: shoulder—town thawed, runs/peaks may keep snow; avoid full-summer flowers."
        if m in {"jun","jul","aug"}: return "Season: summer—no town snow; green valleys."
        if m=="sep": return "Season: early fall—minimal/no snow except high peaks."
        if m=="oct": return "Season: fall—aspens turning; no active skiing unless requested."
    if site=="bostonvacay.com" and m:
        if m in {"dec","jan","feb"}: return "Season: Boston winter; snow optional."
        if m in {"jun","jul","aug"}: return "Season: summer by harbor/Charles."
        if m in {"sep","oct"}: return "Season: fall foliage tones."
        if m in {"apr","may"}: return "Season: spring blooms."
    if site=="bangkokvacay.com": return "Season: tropical warm; no winter scenes."
    return None

ROUTE_SW_KEYWORDS = ["albuquerque","santa fe","taos","new mexico","nm","rio grande","gorge","mesa","pueblo","adobe","sand dunes"]
def parse_route_endpoints(k: str) -> Optional[Tuple[str,str]]:
    k = " ".join(k.lower().split())
    m = re.search(r"between\s+(.+?)\s+and\s+(.+)$", k)
    if m: return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"from\s+(.+?)\s+to\s+(.+)$", k)
    if m: return m.group(1).strip(), m.group(2).strip()
    return None

def corridor_hint_for_route(site: str, keyword: str) -> Optional[str]:
    ep = parse_route_endpoints(keyword)
    if not ep: return None
    k = keyword.lower()
    generic = ("Road-trip corridor scene: highway/overlook or dash+map angle. Brown park sign/mile marker in soft focus. "
               "No brand signage; avoid resort artifacts like gondolas/base lodges.")
    if any(sw in k for sw in ROUTE_SW_KEYWORDS) or ("vail" in k and "albuquerque" in k):
        return (generic + " Southwest palette: sage/piñon-juniper, red/orange mesas, adobe hints; "
                "icons like Rio Grande Gorge Bridge or Great Sand Dunes distant. No alpine gondolas.")
    return generic

# ---------- Planner & Critic ----------
PLANNER_BASE = """
You are a senior creative director writing photorealistic image briefs for a travel/consumer blog.
Return ONE concise prompt (1–2 sentences). Editorial stock-photo style; balanced composition; natural light; {ORIENTATION_NOTE}.
No readable logos or typography. Privacy-respecting. Family-safe.

Nudges:
- "what county is …": tabletop regional map with a pin; county courthouse exterior with seal out of focus (no readable text).
- "what river runs through …": make the river the hero.
- "how far/between/from …": road-trip planning vibe; optional dash+map; no brand signage.
"""

SITE_NUDGES = {
    "vailvacay.com":[
        "If topic implies skiing (back bowls/lift tickets), show snowy slopes and lift infrastructure.",
        "For price/ticket topics: conceptual scene (blurred board/phone); NEVER readable numbers."
    ],
    "bostonvacay.com":[
        "Ferry/Bar Harbor: terminal/ramp with vessel; signage generic/debranded.",
    ],
    "bangkokvacay.com":[
        "BTS/Chinatown: platform/train arrival; wayfinding shapes suggestive; avoid readable text.",
    ],
}

CRITIC_SYSTEM = """
You are a prompt critic. Ensure the prompt clearly illustrates the keyword, honors season/corridor directives,
keeps brand-neutrality (no readable logos/text), and is concise. For price/ticket topics, forbid readable numbers.
If improvement is needed, return {"action":"refine","prompt":"..."}; else {"action":"ok"}.
"""

def build_planner_system(site_key: str, facts_hint: Optional[str], ref_cues: List[str],
                         season_hint: Optional[str], corridor_hint: Optional[str], orientation_note: str) -> str:
    nudges = SITE_NUDGES.get(site_key, [])
    site_block = "\nSite nudges:\n- " + "\n- ".join(nudges) + "\n" if nudges else ""
    facts_block = f"\nContext (do NOT render text/numbers):\n- {facts_hint}\n" if facts_hint else ""
    season_block = f"\nSeason directive (MUST follow):\n- {season_hint}\n" if season_hint else ""
    corridor_block = f"\nCorridor directive (MUST follow):\n- {corridor_hint}\n" if corridor_hint else ""
    cues_block = ("\nReference cues (inspiration only; no text/logos):\n- "+"\n- ".join(ref_cues[:6])+"\n") if ref_cues else ""
    return (PLANNER_BASE.replace("{ORIENTATION_NOTE}", orientation_note)
            + site_block + season_block + corridor_block + facts_block + cues_block
            + '\nOutput JSON ONLY:\n{"prompt":"<one or two sentences>"}')

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
    content = chat_completion(api_key, [{"role":"system","content":CRITIC_SYSTEM},
                                        {"role":"user","content":f"Keyword: {keyword}\nProposed prompt: {prompt}"}],
                              temperature=0.2)
    data = extract_json(content)
    if data and data.get("action")=="refine" and data.get("prompt"):
        return data["prompt"]
    return prompt

# ---------- LSI ----------
def generate_lsi_variants(api_key: str, main_keyword: str, site_key: str, count: int) -> List[str]:
    if count <= 0: return []
    sys = ("You are an SEO strategist. Create concise LSI variants for distinct images. "
           "Keep month/season/location context. No brands. JSON: {\"variants\":[\"...\"]}")
    user = f"Main keyword: {main_keyword}\nSite: {site_key}\nCount: {count}"
    resp = chat_completion(api_key, [{"role":"system","content":sys},{"role":"user","content":user}], temperature=0.5)
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

# ---------- Place/venue detection ----------
VENUE_HINTS = ["restaurant","tavern","bar","rooftop","cafe","hotel","lodge","inn","resort","mall","market","plaza",
               "station","terminal","ferry","pier","museum","gallery","shop","boutique","store","temple","shrine"]
def looks_like_place_query(keyword: str) -> bool:
    k = keyword.lower()
    if any(h in k for h in VENUE_HINTS): return True
    words = re.findall(r"[A-Za-z][A-Za-z']+", keyword)
    caps = sum(1 for w in words if w[:1].isupper())
    return caps >= 3

def build_cc_queries(keyword: str, site_key: str, resolved: Optional[dict]) -> List[str]:
    base_city = {
        "vailvacay.com": "Vail Colorado",
        "bostonvacay.com": "Boston",
        "bangkokvacay.com": "Bangkok",
    }.get(site_key, "")
    q0 = f"{keyword} {base_city}".strip()
    variants = [q0, f"{keyword} exterior {base_city}".strip(), f"{keyword} storefront {base_city}".strip()]
    if resolved:
        name = resolved.get("name") or ""
        if name and name.lower() not in q0.lower():
            variants.insert(0, f"{name} {base_city}".strip())
        for a in resolved.get("alt_names", []):
            variants.append(f"{a} {base_city}".strip())
            variants.append(f"{a} rooftop {base_city}".strip())
    seen, out = set(), []
    for q in variants:
        q = q.strip()
        if q and q not in seen:
            out.append(q); seen.add(q)
    return out

def choose_render_size_for_orientation(chosen_size: str, orientation_note: str) -> str:
    portrait = "portrait" in orientation_note
    if portrait and chosen_size != "1024x1536": return "1024x1536"
    if (not portrait) and chosen_size == "1024x1536": return "1536x1024"
    return chosen_size

# ---------- UI ----------
st.title(f"{APP_NAME} — {APP_VERSION}")
st.caption("CC Preview from Google Images, **Flickr (text + geo)** and **Openverse**, "
           "plus AI replica fallback. Place Resolver uses SerpAPI Maps to find canonical names + coordinates.")

if APP_PASSWORD:
    if st.text_input("Team password", type="password") != APP_PASSWORD:
        st.stop()

with st.sidebar:
    site = st.selectbox("Site", list(SITE_PRESETS.keys()),
                        index=list(SITE_PRESETS.keys()).index(DEFAULT_SITE))
    out_preset = st.selectbox("Output preset", list(OUTPUT_PRESETS.keys()),
                              index=list(OUTPUT_PRESETS.keys()).index(DEFAULT_OUTPUT_PRESET))
    images_per_kw = st.number_input("Images per keyword", 1, 12, 1, 1)
    quality = st.slider("WebP quality", 60, 95, DEFAULT_QUALITY)
    render_size = st.selectbox("Model render size", API_IMAGE_SIZE_OPTIONS, index=0)

    source_mode = st.radio("Source mode",
                           ["Hands-free Real/Replica (CC first)", "AI only", "Real-only CC (skip if none)"])

    st.markdown("**Real photo sources (CC)**")
    serpapi_key = st.text_input("SERPAPI_API_KEY", type="password", value=SERPAPI_DEFAULT,
                                help="Google Images (CC) + Maps resolver.")
    flickr_key  = st.text_input("FLICKR_API_KEY", type="password", value=FLICKR_DEFAULT,
                                help="Flickr (CC) — optional.")
    use_openverse = st.checkbox("Use Openverse (free CC)", value=True)

    use_maps_resolver = st.checkbox("Use Place Resolver (SerpAPI Maps)", value=True,
                                    help="Gets venue’s canonical name + lat/lon; improves Flickr geo matches.")
    preview_cc = st.checkbox("Preview CC candidates (manual pick)", value=True)

openai_key = st.text_input("OpenAI API key", type="password", value=st.secrets.get("OPENAI_API_KEY",""))
keywords_text = st.text_area("Keywords (one per line)", height=220,
                             placeholder="Paradise Lost Bangkok – Rooftop Bar\nBest seafood restaurant in Boston")

c1,c2,c3 = st.columns([1,1,1])
btn_search = c1.button("Search & Preview" if preview_cc else "Generate")
btn_render = c2.button("Render Images")
btn_clear  = c3.button("Clear")

if btn_clear:
    st.session_state.clear()
    st.experimental_rerun()

# ---------- Session state ----------
ss = st.session_state
ss.setdefault("work_items", [])     # list of (main_kw, actual_kw, work_id)
ss.setdefault("lsi_record", {})     # main_kw -> [lsi]
ss.setdefault("cc_candidates", {})  # work_id -> [candidate dict]
ss.setdefault("cc_pick", {})        # work_id -> {"mode":"ai/cc/skip","index":int}
ss.setdefault("out_cfg", OUTPUT_PRESETS[out_preset])
ss.setdefault("render_plan_ready", False)

# ---------- AI helper ----------
def do_ai_image(api_key: str, keyword: str, site_key: str,
                out_cfg: dict, chosen_render_size: str, serpapi_key: str):
    site_vibe = SITE_PRESETS.get(site_key, SITE_PRESETS[DEFAULT_SITE])
    W,H = out_cfg["w"], out_cfg["h"]
    orientation_note = out_cfg["orientation_note"]
    render_sz = choose_render_size_for_orientation(chosen_render_size, orientation_note)

    season_hint = season_hint_for_site(site_key, keyword)
    corridor_hint = corridor_hint_for_route(site_key, keyword)

    facts_hint = None
    if serpapi_key and is_price_topic(keyword):
        facts_hint = serpapi_price_hint(serpapi_key, f"{keyword} {site_key.split('.')[0]}")

    # tiny ref cues from Google Images titles (optional)
    ref_cues = []
    try:
        if serpapi_key:
            url = "https://serpapi.com/search.json"
            params = {"engine":"google","q":f"{keyword} {site_key.split('.')[0]}","tbm":"isch","ijn":"0","tbs":"sur:fc","api_key":serpapi_key}
            r = requests.get(url, params=params, timeout=20)
            titles = [it.get("title") for it in (r.json().get("images_results",[])[:5]) if isinstance(it.get("title"),str)]
            if titles:
                sys = ("Summarize 3–6 neutral visual cues from these image titles, avoiding logos/text.\n"
                       "JSON: {\"cues\":[\"...\"]}")
                content = chat_completion(api_key,
                    [{"role":"system","content":sys},{"role":"user","content":"\n".join(titles)}], temperature=0.2)
                data = extract_json(content) or {}
                ref_cues = [str(c)[:100] for c in data.get("cues",[]) if isinstance(c,str)][:6]
    except Exception:
        pass

    base = plan_prompt(api_key, site_vibe, keyword, site_key, facts_hint, ref_cues, season_hint, corridor_hint, orientation_note)
    prompt = critique_and_refine(api_key, keyword, base)

    png = generate_image_bytes(api_key, prompt, render_sz)
    img = Image.open(io.BytesIO(png))
    return prompt, to_webp_bytes(img, W, H, quality), facts_hint, ref_cues

# ---------- Step 1: Search & Preview ----------
def aggregate_cc_candidates(keyword: str, site_key: str) -> List[dict]:
    """
    Merge CC candidates from: Google Images CC, Flickr (text + geo if resolver), Openverse.
    Deduplicate by original_url.
    """
    cands: List[dict] = []
    resolved = None

    # PlaceResolver
    if use_maps_resolver and serpapi_key and looks_like_place_query(keyword):
        resolved = serpapi_place_resolve(serpapi_key, f"{keyword} {site_key.split('.')[0]}")
    queries = build_cc_queries(keyword, site_key, resolved if looks_like_place_query(keyword) else None)

    # Flickr GEO first if we have lat/lon
    if flickr_key and resolved and resolved.get("lat") and resolved.get("lng"):
        cands += flickr_cc_geo(flickr_key, resolved["lat"], resolved["lng"], tags="rooftop,skyline,bar,restaurant,view", radius_km=0.8, max_items=12)

    # Flickr TEXT
    if flickr_key:
        for q in queries:
            cands += flickr_cc_text(flickr_key, q, max_items=6)
            if len(cands) >= 18: break

    # Google Images CC
    if serpapi_key:
        for q in queries:
            cands += serpapi_cc_images(serpapi_key, q, max_items=6)
            if len(cands) >= 24: break

    # Openverse
    if use_openverse:
        for q in queries[:4]:
            cands += openverse_cc_images(q, max_items=6)
            if len(cands) >= 30: break

    # De-dupe by original_url
    seen, uniq = set(), []
    for c in cands:
        u = c.get("original_url")
        if u and u not in seen:
            uniq.append(c); seen.add(u)
    return uniq[:30]

if btn_search:
    out_cfg = OUTPUT_PRESETS[out_preset]; ss["out_cfg"] = out_cfg
    base_kws = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
    if not base_kws:
        st.warning("Please paste at least one keyword."); st.stop()

    work_items = []; lsi_record = {}
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

    if preview_cc and source_mode != "AI only":
        st.success("Searching CC sources…")
        for _main, actual, work_id in work_items:
            cands = aggregate_cc_candidates(actual, site)
            ss["cc_candidates"][work_id] = cands
        st.toast("CC candidates ready. Review below, pick your choice(s), then click Render.", icon="✅")
    else:
        st.info("Skipping CC preview. Click **Render Images** to create output.")
    ss["render_plan_ready"] = True

# ---------- CC Preview UI ----------
if preview_cc and source_mode != "AI only" and ss["work_items"]:
    st.markdown("## CC Candidates (manual pick)")
    for (main_kw, actual_kw, work_id) in ss["work_items"]:
        candidates = ss["cc_candidates"].get(work_id, [])
        if not candidates: continue
        st.markdown(f"**{actual_kw}**")
        labels, values = [], []
        for i,c in enumerate(candidates):
            src = c.get("source","")
            title = (c.get("title") or "").strip()
            host  = (c.get("host") or "").strip()
            labels.append(f"{src} #{i+1} — {title[:70]}{' from ' + host if host else ''}")
            values.append(f"cc:{i}")
        if source_mode == "Real-only CC (skip if none)":
            labels = ["Skip this item"] + labels; values = ["skip"] + values; default="skip"
        else:
            labels = ["AI replica"] + labels; values = ["ai"] + values; default="ai"
        choice = st.radio("Pick one:", options=values, index=0,
                          format_func=lambda v: labels[values.index(v)], key=f"pick_{work_id}")
        ss["cc_pick"][work_id] = {"mode": choice.split(":")[0],
                                  "index": int(choice.split(":")[1]) if choice.startswith("cc:") else -1}

        cols = st.columns(4)
        for i,c in enumerate(candidates[:12]):
            with cols[i % 4]:
                st.image(c.get("thumbnail_url") or c.get("original_url"), caption=f"{c.get('source','')} #{i+1}",
                         use_container_width=True)

# ---------- Step 2: Render ----------
def write_attribution(zipf, lines: List[str]):
    if lines:
        zipf.writestr("ATTRIBUTION.txt", "\n".join(lines))

if btn_render:
    if not ss["work_items"]:
        st.warning("Run **Search & Preview** first (or disable preview)."); st.stop()
    out_cfg = ss["out_cfg"]; W,H = out_cfg["w"], out_cfg["h"]

    prog = st.progress(0); done=0
    thumbs: List[Tuple[str, bytes]] = []
    meta_log: List[str] = []
    prompts_used: List[Tuple[str, str]] = []
    facts_notes: List[Tuple[str, str]] = []
    attribution: List[str] = []

    zip_buf = io.BytesIO(); zipf = zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED)

    for (main_kw, actual_kw, work_id) in ss["work_items"]:
        fname_base = work_id; used_real=False; webp=None; prompt_used=None

        cands = ss["cc_candidates"].get(work_id, [])
        pick  = ss["cc_pick"].get(work_id, {"mode":"ai","index":-1})

        def save_cc(cand: dict):
            nonlocal webp, used_real
            img = safe_download_image(cand.get("original_url",""))
            if not img: return False
            webp = to_webp_bytes(img, W, H, quality); used_real=True
            src = cand.get("source","")
            if src=="Flickr":
                line = f"{fname_base} -- Flickr CC: \"{cand.get('title','')}\" by {cand.get('owner_name','')} "\
                       f"({cand.get('license','')}) {cand.get('license_url','')} | {cand.get('attribution_url','')}"
            elif src=="Openverse":
                line = f"{fname_base} -- Openverse: \"{cand.get('title','')}\" by {cand.get('owner_name','')} "\
                       f"({cand.get('license','')}) {cand.get('license_url','')} | {cand.get('attribution_url','')}"
            else: # GoogleCC
                line = f"{fname_base} -- Google Images (CC as reported): {cand.get('original_url','')} | "\
                       f"Title: {cand.get('title','')} | Source: {cand.get('host','')}"
            attribution.append(line)
            return True

        if preview_cc and source_mode!="AI only" and cands:
            mode = pick["mode"]; idx = pick["index"]
            if mode=="cc" and 0<=idx<len(cands):
                if save_cc(cands[idx]): meta_log.append(f"{actual_kw} → CC photo (manual pick)")
                else: meta_log.append(f"{actual_kw} → chosen CC failed; using AI")

            elif mode=="skip":
                meta_log.append(f"{actual_kw} → skipped by user")
                done += 1; prog.progress(done/len(ss["work_items"])); continue

        if not used_real and source_mode!="AI only":
            # hands-free try: aggregate and auto-pick first best
            auto = aggregate_cc_candidates(actual_kw, site)
            if auto:
                if save_cc(auto[0]): meta_log.append(f"{actual_kw} → CC photo (auto)")
        if not used_real:
            if source_mode=="Real-only CC (skip if none)":
                meta_log.append(f"{actual_kw} → skipped (no CC found)")
                done += 1; prog.progress(done/len(ss["work_items"])); continue
            if not openai_key:
                st.error("OpenAI key required for AI replica."); st.stop()
            try:
                prompt_used, webp, facts_hint, ref_cues = do_ai_image(openai_key, actual_kw, site, out_cfg, render_size, serpapi_key)
                prompts_used.append((f"{fname_base}.webp", prompt_used))
                if facts_hint: facts_notes.append((fname_base, facts_hint))
                meta_log.append(f"{actual_kw} → AI replica")
            except Exception as e:
                st.error(f"{actual_kw}: {e}")
                done += 1; prog.progress(done/len(ss["work_items"])); continue

        if webp:
            suffix = "photo-cc" if used_real else "replica"
            out_name = f"{fname_base}--{suffix}.webp"
            zipf.writestr(out_name, webp)
            thumbs.append((out_name, webp))

        done += 1; prog.progress(done/len(ss["work_items"]))

    write_attribution(zipf, attribution)
    zipf.close(); zip_buf.seek(0)

    if not thumbs: st.error("No images were produced.")
    else:
        st.success("Done! Download your images below.")
        st.download_button("⬇️ Download ZIP", data=zip_buf,
                           file_name=f"{slugify(site)}_{slugify(out_preset)}.zip", mime="application/zip")
        st.markdown("### Previews")
        cols = st.columns(3 if H<=W else 2)
        for i,(name,data_bytes) in enumerate(thumbs):
            with cols[i % len(cols)]:
                st.image(data_bytes, caption=name, use_container_width=True)
                st.download_button("Download", data=data_bytes, file_name=name, mime="image/webp", key=f"dl_{i}")

        if meta_log:
            with st.expander("Run log"): 
                for line in meta_log: st.write("• " + line)
        if ss["lsi_record"]:
            with st.expander("LSI variants generated"):
                for main_kw, variants in ss["lsi_record"].items():
                    st.markdown(f"**{main_kw}**"); st.write(", ".join(variants) if variants else "_None_")
        if prompts_used:
            with st.expander("Prompts used (AI)"):
                for fname,p in prompts_used:
                    st.markdown(f"**{fname}**"); st.code(p, language="text")
