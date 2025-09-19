# ImageForge Autopilot v2.6.0 — Pinterest preset
# Adds an output preset for Pinterest Pins (1000×1500, 2:3) + portrait-oriented prompts.
# Keeps: Season Engine, Corridor Engine, SerpAPI (optional), ZIP + per-image downloads.
#
# Requirements: streamlit, requests, pillow
# Optional Secrets: OPENAI_API_KEY, SERPAPI_API_KEY, APP_PASSWORD

import io
import re
import json
import base64
import zipfile
from typing import List, Tuple, Optional

import requests
from PIL import Image
import streamlit as st

# =========================
# App identity & config
# =========================
APP_NAME = "ImageForge Autopilot"
APP_VERSION = "v2.6.0 — Pinterest preset"

st.set_page_config(page_title=f"{APP_NAME} — {APP_VERSION}", layout="wide")

# Optional: password gate & default site
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")
DEFAULT_SITE = "vailvacay.com"

# Site vibes (brand-safe)
SITE_PRESETS = {
    "vailvacay.com":  "Colorado alpine resort & village; riverside paths, gondola/base areas, evergreen forests, cozy lodges.",
    "bostonvacay.com":"New England city; Beacon Hill brick, harborwalk, Charles River, landmark stations.",
    "bangkokvacay.com":"Bangkok; temples, canals, night markets, BTS/MRT platforms, bustling street life.",
    "ipetzo.com":      "Pet lifestyle; dogs/cats with owners in parks or cozy homes; neutral interiors.",
    "1-800deals.com":  "Retail/e-commerce; parcels, unboxing, generic products; clean backgrounds.",
}

# DALLE render sizes allowed
API_IMAGE_SIZE_OPTIONS = ["1536x1024", "1024x1024", "1024x1536"]

# Output presets (final WebP)
OUTPUT_PRESETS = {
    "Blog 1200×675 (16:9)": {"w": 1200, "h": 675, "orientation_note": "landscape orientation"},
    "Pinterest 1000×1500 (2:3)": {"w": 1000, "h": 1500, "orientation_note": "portrait/vertical orientation"},
}
DEFAULT_OUTPUT_PRESET = "Blog 1200×675 (16:9)"
DEFAULT_QUALITY = 82

# =========================
# Utilities
# =========================
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
        new_w = int(H * tr)
        left = (W - new_w) // 2
        box = (left, 0, left + new_w, H)
    else:
        new_h = int(W / tr)
        top = (H - new_h) // 2
        box = (0, top, W, top + new_h)
    return img.crop(box)

def to_webp_bytes(img: Image.Image, w: int, h: int, quality: int) -> bytes:
    img = img.convert("RGB")
    img = crop_to_aspect(img, w, h).resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=quality, method=6)
    return buf.getvalue()

def extract_json(txt: str):
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

# =========================
# OpenAI helpers
# =========================
def chat_completion(api_key: str, messages: list, temperature: float = 0.7, model: str = "gpt-4o-mini") -> str:
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
        img = requests.get(d0["url"], timeout=180)
        img.raise_for_status()
        return img.content
    raise RuntimeError("No image data returned")

# =========================
# SerpAPI helpers (optional)
# =========================
SERPAPI_DEFAULT = st.secrets.get("SERPAPI_API_KEY", "")

PRICE_TERMS = [
    "price", "prices", "cost", "how much", "ticket", "tickets", "lift ticket",
    "gondola ticket", "fee", "fees", "discount", "coupon"
]

def keyword_is_pricey(k: str) -> bool:
    kk = k.lower()
    return any(term in kk for term in PRICE_TERMS)

def serpapi_price_hint(api_key: str, query: str) -> Optional[str]:
    try:
        url = "https://serpapi.com/search.json"
        params = {"engine": "google", "q": query, "num": 5, "api_key": api_key}
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()

        texts = []
        ab = data.get("answer_box") or {}
        for k in ("snippet", "title", "answer"):
            v = ab.get(k)
            if isinstance(v, str):
                texts.append(v)
        for res in data.get("organic_results", [])[:5]:
            for k in ("snippet", "title"):
                v = res.get(k)
                if isinstance(v, str):
                    texts.append(v)

        blob = " ".join(texts)[:1000]
        m = re.findall(r"\$\s*\d{2,4}", blob)
        if m:
            uniq = sorted(set(s.replace(" ", "") for s in m))
            joined = ", ".join(uniq[:4])
            return f"Public search hints show figures like: {joined}. Treat as reference only; do not render text or numbers."
        if blob:
            return "Public search hints retrieved. Treat as reference only; do not render text or numbers."
        return None
    except Exception:
        return None

def serpapi_image_titles(api_key: str, query: str, cc_only: bool = True, max_items: int = 6) -> List[str]:
    try:
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google",
            "q": query,
            "tbm": "isch",
            "ijn": "0",
            "api_key": api_key
        }
        if cc_only:
            params["tbs"] = "sur:fc"
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()
        results = data.get("images_results", [])[:max_items]
        titles = []
        for it in results:
            t = it.get("title")
            if isinstance(t, str) and t.strip():
                titles.append(t.strip())
        return titles
    except Exception:
        return []

def summarize_titles_to_cues(openai_key: str, keyword: str, titles: List[str]) -> List[str]:
    if not titles:
        return []
    sys = (
        "You are a visual summarizer. Given a keyword and a set of image titles from a public image search, "
        "extract 3–6 neutral visual cues that would help illustrate the topic without copying or rendering text. "
        "Avoid brands, readable signage, and explicit content. "
        "Return JSON ONLY: {\"cues\": [\"cue1\", \"cue2\", ...]}"
    )
    user = f"Keyword: {keyword}\nTitles:\n- " + "\n- ".join(titles)
    content = chat_completion(openai_key, [
        {"role": "system", "content": sys},
        {"role": "user", "content": user}
    ], temperature=0.2)
    data = extract_json(content)
    if not data or "cues" not in data or not isinstance(data["cues"], list):
        return []
    cues = [str(c)[:100] for c in data["cues"] if isinstance(c, str)]
    return cues[:6]

# =========================
# Season Engine
# =========================
MONTH_ALIASES = {
    "january":"jan", "february":"feb", "march":"mar", "april":"apr", "may":"may", "june":"jun",
    "july":"jul", "august":"aug", "september":"sep", "october":"oct", "november":"nov", "december":"dec",
    "jan":"jan","feb":"feb","mar":"mar","apr":"apr","jun":"jun","jul":"jul","aug":"aug","sep":"sep","oct":"oct","nov":"nov","dec":"dec"
}

def extract_month_token(text: str) -> Optional[str]:
    t = text.lower()
    for word in re.findall(r"[a-z]+", t):
        if word in MONTH_ALIASES:
            return MONTH_ALIASES[word]
    return None

def skiish(keyword: str) -> bool:
    k = keyword.lower()
    return any(s in k for s in [
        "ski", "back bowl", "lift ticket", "gondola", "powder", "snowboard", "chairlift"
    ])

def season_hint_for_site(site_key: str, keyword: str) -> Optional[str]:
    m = extract_month_token(keyword)
    k = keyword.lower()

    if site_key == "vailvacay.com":
        if skiish(k):
            return ("Season: winter skiing. Show snowy runs and operating lifts; dressed-for-winter visitors. "
                    "No summer flowers or green meadows.")
        if m in {"nov","dec","jan","feb","mar","apr"}:
            if m == "apr":
                return ("Season: late-season spring skiing. Snowy ski runs with skiers; lifts operating; "
                        "sunny is fine but avoid summer wildflowers and lush green meadows.")
            return ("Season: winter. Snow on ski slopes, winter clothing, lift infrastructure active.")
        if m == "may":
            return ("Season: shoulder. Some snow on high peaks is fine, town mostly thawed; "
                    "avoid full-summer flowers; if activity relates to skiing, keep snow on runs.")
        if m in {"jun","jul","aug"}:
            return ("Season: summer. No snow in town; high peaks may have small patches only.")
        if m in {"sep"}:
            return ("Season: early fall. Greens fading; minimal or no snow except high peaks.")
        if m in {"oct"}:
            return ("Season: fall with aspens turning; possible light snow dusting on peaks; no active skiing unless keyword says so.")
        if "winter" in k:
            return "Season: winter. Snowy slopes and winter clothing."
        if "summer" in k:
            return "Season: summer. No snow in town."
        if "spring" in k:
            return "Season: spring in mountains; snow may persist on runs/high peaks."

    if site_key == "bostonvacay.com" and m:
        if m in {"dec","jan","feb"}:
            return "Season: Boston winter; cold-weather outfits; snow streetscape optional."
        if m in {"jun","jul","aug"}:
            return "Season: summer by the harbor/Charles; leafy trees; no snow."
        if m in {"sep","oct"}:
            return "Season: fall; foliage tones; jackets."
        if m in {"apr","may"}:
            return "Season: spring; light jackets; blooming trees."

    if site_key == "bangkokvacay.com":
        return "Season: tropical warm; no coats or winter scenes."

    return None

# =========================
# Corridor Engine
# =========================
ROUTE_SW_KEYWORDS = [
    "albuquerque", "santa fe", "taos", "new mexico", "nm",
    "rio grande", "gorge", "mesa", "pueblo", "adobe", "sand dunes", "great sand dunes"
]

def parse_route_endpoints(keyword: str) -> Optional[Tuple[str, str]]:
    k = " ".join(keyword.lower().split())
    m = re.search(r"between\s+(.+?)\s+and\s+(.+)$", k)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"from\s+(.+?)\s+to\s+(.+)$", k)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None

def corridor_hint_for_route(site_key: str, keyword: str) -> Optional[str]:
    endpoints = parse_route_endpoints(keyword)
    if not endpoints:
        if re.search(r"\bto\b", keyword.lower()) and ("between" in keyword.lower() or "from" in keyword.lower()):
            pass
        else:
            return None

    k = keyword.lower()
    generic = (
        "Road-trip corridor scene: emphasize highway or scenic overlook, or a dash/map angle. "
        "Use brown park sign or mile marker in soft focus. Keep brand signage unreadable. "
        "Avoid resort-only artifacts such as gondolas, chairlifts, and base lodges."
    )
    if any(sw in k for sw in ROUTE_SW_KEYWORDS) or ("vail" in k and "albuquerque" in k):
        return (generic + " Southwest palette: sagebrush and piñon-juniper, red/orange mesas, adobe or Spanish-style town hints; "
                "optional icons like Rio Grande Gorge Bridge or Great Sand Dunes in the distance. No alpine gondolas.")
    return generic

# =========================
# Planner + Critic
# =========================
PLANNER_BASE = """
You are a senior creative director writing photorealistic image briefs
for a travel/consumer blog. Given a keyword and a site vibe, craft ONE concise prompt
(1–2 sentences) for a DALLE-like model. Infer obvious visual cues so the topic is unmistakable.
Use editorial stock-photo style.

Hard constraints (always include as plain text in the prompt):
- No text/typography overlays, no readable logos or brand marks, privacy-respecting, non-explicit.
- Balanced composition, natural light, {ORIENTATION_NOTE}.

Pattern nudges (examples, not outputs):
- "what county is …": tabletop regional map with a pin near the destination OR exterior of county courthouse with seal out of focus (no readable text).
- "what river runs through …": make the river the hero; town context secondary.
- "what mountain range is … in": emphasize ridgelines/peaks; town minimal.
- "how far/between/from A to B": road-trip planning vibe (two-lane highway or scenic overlook; optional dashboard/map angle); avoid readable text or brand signage.
"""

SITE_NUDGES = {
    "vailvacay.com": [
        "Snow topics: if ski/back bowls/lift tickets appear, winter conditions are natural; think snowy slopes and lift infrastructure.",
        "For lift-ticket/gondola price topics: conceptual scene only (blurred board/phone); never readable numbers."
    ],
    "bostonvacay.com": [
        "Ferry/Bar Harbor: boarding ramp or terminal scene with vessel in frame; signage generic/debranded.",
        "Wardrobe seasons: layered outfits appropriate to month; no visible logos.",
        "Birthday freebies: café/pastry counter; POS blurred; no readable signage.",
    ],
    "bangkokvacay.com": [
        "BTS/Chinatown: BTS platform with arriving train; wayfinding shapes/colors suggestive but unreadable; Chinatown lantern hints.",
        "Passport/visa photo: simple studio corner with backdrop + softbox + tripod; no brands.",
        "Nightlife: exterior/neon street scene; silhouettes; non-explicit.",
    ],
    "ipetzo.com": [
        "Dog-friendly: owner with dog on patio/lobby seating; water bowl; no hotel branding.",
        "Indoor pet activities: living-room play scene; no brand marks.",
    ],
    "1-800deals.com": [
        "Unboxing/deals/coupons: generic parcels/product tableaus; device screens out-of-focus; no trademarks.",
    ],
}

def build_planner_system(site_key: str,
                         facts_hint: Optional[str],
                         ref_cues: List[str],
                         season_hint: Optional[str],
                         corridor_hint: Optional[str],
                         orientation_note: str) -> str:
    nudges = SITE_NUDGES.get(site_key, [])
    site_block = "\nSite-specific nudges:\n- " + "\n- ".join(nudges) + "\n" if nudges else ""
    facts_block = f"\nContext (for concept only; do NOT render text or numbers):\n- {facts_hint}\n" if facts_hint else ""
    season_block = f"\nSeason directive (must follow):\n- {season_hint}\n" if season_hint else ""
    corridor_block = f"\nCorridor directive (must follow):\n- {corridor_hint}\n" if corridor_hint else ""
    cues_block = ""
    if ref_cues:
        cues_block = "\nReference cues (from public image titles; inspiration only, do not copy layouts; no text/logos):\n- " + "\n- ".join(ref_cues[:6]) + "\n"

    return (
        PLANNER_BASE.replace("{ORIENTATION_NOTE}", orientation_note)
        + site_block
        + season_block
        + corridor_block
        + facts_block
        + cues_block
        + '\nOutput JSON ONLY:\n{"prompt": "<final one- or two-sentence prompt>"}'
    )

CRITIC_SYSTEM = """
You are a prompt critic. Given a keyword and a proposed image prompt,
decide if the prompt obviously covers the keyword. If missing, revise succinctly.
Keep style constraints intact (no text/logos, non-explicit; balanced composition; obey orientation).
For price/cost/ticket topics, ensure it forbids readable numbers/prices/signage and uses a conceptual scene.
Respect any season directive (e.g., 'late-season spring skiing').
If a corridor directive is present, you MUST depict a road-trip scene (highway/overlook or dash+map angle)
and you MUST avoid resort artifacts such as gondolas, chairlifts, base lodges; add regional palette if specified.
Use provided reference cues only as inspiration; do not copy layouts or text.
Output JSON ONLY:
- If OK: {"action":"ok"}
- If needs fix: {"action":"refine","prompt":"<better prompt>"}
"""

def plan_prompt(api_key: str, site_vibe: str, keyword: str, site_key: str,
                facts_hint: Optional[str], ref_cues: List[str],
                season_hint: Optional[str], corridor_hint: Optional[str],
                orientation_note: str) -> str:
    planner_system = build_planner_system(site_key, facts_hint, ref_cues, season_hint, corridor_hint, orientation_note)
    user = f"""Site vibe: {site_vibe}
Keyword: {keyword}

Write one photorealistic travel-blog prompt that:
- Clearly signals the topic with obvious visual cues
- Chooses indoor vs outdoor and season naturally (and obeys directives if provided)
- Uses road-trip visuals for corridor queries
- Stays safe and brand-neutral
- Is concise (max ~2 sentences)"""
    content = chat_completion(api_key, [
        {"role":"system","content":planner_system},
        {"role":"user","content":user}
    ])
    data = extract_json(content)
    if not data or "prompt" not in data:
        return content
    return data["prompt"]

def critique_and_refine(api_key: str, keyword: str, prompt: str) -> str:
    user = f"Keyword: {keyword}\nProposed prompt: {prompt}"
    content = chat_completion(api_key, [
        {"role":"system","content":CRITIC_SYSTEM},
        {"role":"user","content":user}
    ], temperature=0.2)
    data = extract_json(content)
    if not data:
        return prompt
    if data.get("action") == "refine" and data.get("prompt"):
        return data["prompt"]
    return prompt

# =========================
# UI
# =========================
st.title(f"{APP_NAME} — {APP_VERSION}")
st.caption("Choose an output preset (Blog or Pinterest). Pinterest uses a 2:3 vertical crop and portrait-oriented prompts. "
           "Season Engine fixes month visuals; Corridor Engine fixes 'between/from A to B' queries. "
           "Optional SerpAPI adds reference cues and price hints. Output: WebP + ZIP + per-image downloads.")

if APP_PASSWORD:
    gate = st.text_input("Team password", type="password")
    if gate != APP_PASSWORD:
        st.stop()

with st.sidebar:
    site = st.selectbox("Site", list(SITE_PRESETS.keys()),
                        index=list(SITE_PRESETS.keys()).index(DEFAULT_SITE))
    output_preset = st.selectbox("Output preset", list(OUTPUT_PRESETS.keys()),
                                 index=list(OUTPUT_PRESETS.keys()).index(DEFAULT_OUTPUT_PRESET))
    quality = st.slider("WebP quality", 60, 95, DEFAULT_QUALITY)
    render_size = st.selectbox(
        "Model render size (internal)",
        API_IMAGE_SIZE_OPTIONS,
        index=0,
        help="For Pinterest, a vertical render (1024×1536) gives best detail; we auto-switch if needed."
    )
    serpapi_key = st.text_input("SERPAPI_API_KEY (optional)", type="password",
                                value=SERPAPI_DEFAULT, help="Adds reference cues and price hints.")

openai_key = st.text_input(
    "OpenAI API key",
    type="password",
    value=st.secrets.get("OPENAI_API_KEY", ""),
    help="You can also set OPENAI_API_KEY in Streamlit Secrets."
)

keywords_text = st.text_area(
    "Keywords (one per line)",
    height=280,
    placeholder="things to do in Vail in April\nbest seafood restaurant in Boston\nwhat county is Vail\nthings to see between Vail and Albuquerque",
)

c1, c2 = st.columns(2)
run = c1.button("Generate")
if c2.button("Clear"):
    st.session_state.clear()
    st.experimental_rerun()

# =========================
# Run
# =========================
def choose_render_size_for_orientation(chosen_size: str, orientation_note: str) -> str:
    portrait = "portrait" in orientation_note
    if portrait and chosen_size != "1024x1536":
        return "1024x1536"
    if (not portrait) and chosen_size == "1024x1536":
        return "1536x1024"
    return chosen_size

def do_one(api_key: str, keyword: str, site_key: str,
           output_cfg: dict, chosen_render_size: str, quality: int,
           serpapi_key: str) -> Tuple[str, bytes, Optional[str], List[str], List[str]]:
    """
    Returns: (final_prompt, webp_bytes, facts_hint, ref_titles, ref_cues)
    """
    site_vibe = SITE_PRESETS.get(site_key, SITE_PRESETS[DEFAULT_SITE])
    W, H = output_cfg["w"], output_cfg["h"]
    orientation_note = output_cfg["orientation_note"]

    # Harmonize render size with orientation
    render_size = choose_render_size_for_orientation(chosen_render_size, orientation_note)

    # Season directive
    season_hint = season_hint_for_site(site_key, keyword)

    # Corridor directive
    corridor_hint = corridor_hint_for_route(site_key, keyword)

    # SerpAPI price facts (only for price-like queries)
    facts_hint = None
    if serpapi_key and keyword_is_pricey(keyword):
        q = f"{keyword} {site_key.split('.')[0]}"
        facts_hint = serpapi_price_hint(serpapi_key, q)

    # SerpAPI reference cues (image titles -> cues)
    ref_titles: List[str] = []
    ref_cues: List[str] = []
    if serpapi_key:
        q = f"{keyword} {site_key.split('.')[0]}"
        ref_titles = serpapi_image_titles(serpapi_key, q, cc_only=True, max_items=5)
        ref_cues = summarize_titles_to_cues(api_key, keyword, ref_titles) if ref_titles else []

    # Planner
    base_prompt = plan_prompt(
        api_key, site_vibe, keyword, site_key,
        facts_hint, ref_cues, season_hint, corridor_hint,
        orientation_note
    )

    # Critic (single pass)
    final_prompt = critique_and_refine(api_key, keyword, base_prompt)

    # Image
    png = generate_image_bytes(api_key, final_prompt, render_size)
    img = Image.open(io.BytesIO(png))
    return final_prompt, to_webp_bytes(img, W, H, quality), facts_hint, ref_titles, ref_cues

if run:
    if not openai_key:
        st.warning("Please enter your OpenAI API key.")
        st.stop()

    kws: List[str] = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
    if not kws:
        st.warning("Please paste at least one keyword.")
        st.stop()

    out_cfg = OUTPUT_PRESETS[output_preset]

    prog = st.progress(0)
    status = st.empty()
    thumbs: List[Tuple[str, bytes]] = []
    prompts_used: List[Tuple[str, str]] = []
    facts_notes: List[Tuple[str, str]] = []
    ref_notes: List[Tuple[str, List[str], List[str]]] = []

    zip_buf = io.BytesIO()
    zipf = zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED)

    for i, kw in enumerate(kws, start=1):
        fname = f"{slugify(kw)}.webp"
        try:
            status.text(f"Generating {i}/{len(kws)}: {kw}")
            prompt_used, webp, facts_hint, ref_titles, ref_cues = do_one(
                openai_key, kw, site, out_cfg, render_size, quality, serpapi_key
            )
            zipf.writestr(fname, webp)
            thumbs.append((fname, webp))
            prompts_used.append((fname, prompt_used))
            if facts_hint:
                facts_notes.append((fname, facts_hint))
            if ref_titles or ref_cues:
                ref_notes.append((fname, ref_titles, ref_cues))
        except Exception as e:
            st.error(f"{kw}: {e}")
        prog.progress(i/len(kws))

    zipf.close()
    zip_buf.seek(0)

    if not thumbs:
        st.error("No images were generated.")
    else:
        st.success("Done! Download your images below.")
        st.download_button("⬇️ Download ZIP", data=zip_buf,
                           file_name=f"{slugify(site)}_{slugify(output_preset)}.zip",
                           mime="application/zip", key="zip")

        st.markdown("### Previews")
        cols = st.columns(3 if out_cfg['h'] <= out_cfg['w'] else 2)
        for idx, (fname, data_bytes) in enumerate(thumbs):
            with cols[idx % len(cols)]:
                st.image(data_bytes, caption=fname, use_container_width=True)
                st.download_button("Download", data=data_bytes, file_name=fname,
                                   mime="image/webp", key=f"dl_{idx}")

        with st.expander("Prompts used (QA)"):
            for fname, p in prompts_used:
                st.markdown(f"**{fname}**")
                st.code(p, language="text")

        if facts_notes:
            with st.expander("Price facts assist (reference only)"):
                for fname, note in facts_notes:
                    st.markdown(f"**{fname}**")
                    st.write(note)

        if ref_notes:
            with st.expander("Reference cues (public image titles)"):
                st.write("Used as inspiration only; no external images were downloaded.")
                for fname, titles, cues in ref_notes:
                    st.markdown(f"**{fname}**")
                    if titles:
                        st.markdown("- Raw titles:")
                        st.write("; ".join(titles))
                    if cues:
                        st.markdown("- Summarized cues used:")
                        st.write(", ".join(cues))
