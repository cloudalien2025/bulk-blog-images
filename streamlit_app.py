# ImageForge Autopilot v2.4.1 — Season-aware
# 1200×675 WebP bulk generator with:
# - Planner + Critic prompts
# - Optional SerpAPI reference cues (image titles only) + price facts hint
# - NEW: Season Engine (month words -> site-specific visuals; Vail April => late-season skiing)
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
APP_VERSION = "v2.4.1 — Season-aware"

st.set_page_config(page_title=f"{APP_NAME} — {APP_VERSION}", layout="wide")

# Optional: password gate & default site
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")
DEFAULT_SITE = "vailvacay.com"

# Minimal, brand-safe site vibes (tone/place cues only)
SITE_PRESETS = {
    "vailvacay.com":  "Colorado alpine resort & village; riverside paths, gondola/base areas, evergreen forests, cozy lodges.",
    "bostonvacay.com":"New England city; Beacon Hill brick, harborwalk, Charles River, landmark stations.",
    "bangkokvacay.com":"Bangkok; temples, canals, night markets, BTS/MRT platforms, bustling street life.",
    "ipetzo.com":      "Pet lifestyle; dogs/cats with owners in parks or cozy homes; neutral interiors.",
    "1-800deals.com":  "Retail/e-commerce; parcels, unboxing, generic products; clean backgrounds.",
}

API_IMAGE_SIZE_OPTIONS = ["1536x1024", "1024x1024", "1024x1536"]  # supported by gpt-image-1
OUT_W, OUT_H = 1200, 675
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
# Season Engine (NEW)
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
    """Return a strong season directive if we can infer it from the month and site."""
    m = extract_month_token(keyword)
    k = keyword.lower()

    # Vail-focused rules (most critical)
    if site_key == "vailvacay.com":
        # If ski-ish at any time, force winter scene
        if skiish(k):
            return ("Season: winter skiing. Show snowy runs and operating lifts; dressed-for-winter visitors. "
                    "No summer flowers or green meadows.")
        if m in {"nov","dec","jan","feb","mar","apr"}:
            # April tends to be spring skiing; still snow on runs.
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
        # No month, but if keyword mentions "winter", "summer", etc.
        if "winter" in k:
            return "Season: winter. Snowy slopes and winter clothing."
        if "summer" in k:
            return "Season: summer. No snow in town."
        if "spring" in k:
            return "Season: spring in mountains; snow may persist on runs/high peaks."

    # Boston rough rules (optional)
    if site_key == "bostonvacay.com" and m:
        if m in {"dec","jan","feb"}:
            return "Season: Boston winter; cold-weather outfits; snow streetscape optional."
        if m in {"jun","jul","aug"}:
            return "Season: summer by the harbor/Charles; leafy trees; no snow."
        if m in {"sep","oct"}:
            return "Season: fall; foliage tones; jackets."
        if m in {"apr","may"}:
            return "Season: spring; light jackets; blooming trees."

    # Bangkok (always warm)
    if site_key == "bangkokvacay.com":
        return "Season: tropical warm; no coats or winter scenes."

    return None

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
- Balanced composition, natural light, landscape orientation.

Pattern nudges (examples, not outputs):
- "what county is …": tabletop regional map with a pin near the destination OR exterior of county courthouse with seal out of focus (no readable text).
- "what river runs through …": make the river the hero; town context secondary.
- "what mountain range is … in": emphasize ridgelines/peaks; town minimal.
- "how far is … from …": travel-planning vibe (map with two pinned points or dashboard/GPS angle); avoid readable text.

Price/cost/ticket keywords:
- Depict a conceptual price-checking or planning scene (ticket window with blurred board, phone with out-of-focus checkout page, pass + gloves on a counter).
- Absolutely do not render readable numbers, prices, or legible signage.
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

def build_planner_system(site_key: str, facts_hint: Optional[str], ref_cues: List[str], season_hint: Optional[str]) -> str:
    nudges = SITE_NUDGES.get(site_key, [])
    site_block = "\nSite-specific nudges:\n- " + "\n- ".join(nudges) + "\n" if nudges else ""
    facts_block = f"\nContext (for concept only; do NOT render text or numbers):\n- {facts_hint}\n" if facts_hint else ""
    season_block = f"\nSeason directive (must follow):\n- {season_hint}\n" if season_hint else ""
    cues_block = ""
    if ref_cues:
        cues_block = "\nReference cues (from public image titles; inspiration only, do not copy layouts; no text/logos):\n- " + "\n- ".join(ref_cues[:6]) + "\n"
    return (
        PLANNER_BASE
        + site_block
        + season_block
        + facts_block
        + cues_block
        + '\nOutput JSON ONLY:\n{"prompt": "<final one- or two-sentence prompt>"}'
    )

CRITIC_SYSTEM = """
You are a prompt critic. Given a keyword and a proposed image prompt,
decide if the prompt obviously covers the keyword. If missing, revise succinctly.
Keep style constraints intact (no text/logos, non-explicit; balanced composition; landscape).
For price/cost/ticket topics, ensure it forbids readable numbers/prices/signage and uses a conceptual scene.
Respect any explicit season directive (e.g., 'late-season spring skiing' implies snowy ski runs and operating lifts; avoid summer wildflowers).
Use provided reference cues only as inspiration; do not copy layouts or text.
Output JSON ONLY:
- If OK: {"action":"ok"}
- If needs fix: {"action":"refine","prompt":"<better prompt>"}
"""

def plan_prompt(api_key: str, site_vibe: str, keyword: str, site_key: str,
                facts_hint: Optional[str], ref_cues: List[str], season_hint: Optional[str]) -> str:
    planner_system = build_planner_system(site_key, facts_hint, ref_cues, season_hint)
    user = f"""Site vibe: {site_vibe}
Keyword: {keyword}

Write one photorealistic travel-blog prompt that:
- Clearly signals the topic with obvious visual cues
- Chooses indoor vs outdoor and season naturally (and obeys the season directive if provided)
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
st.caption("Paste keywords (one per line). Optional SerpAPI adds reference cues (image titles only) and price hints. "
           "The Season Engine infers month from the keyword (e.g., 'April in Vail') and enforces the right visuals "
           "(e.g., snowy late-season skiing in April). Output: 1200×675 WebP, ZIP + per-image downloads.")

if APP_PASSWORD:
    gate = st.text_input("Team password", type="password")
    if gate != APP_PASSWORD:
        st.stop()

with st.sidebar:
    site = st.selectbox("Site", list(SITE_PRESETS.keys()),
                        index=list(SITE_PRESETS.keys()).index(DEFAULT_SITE))
    quality = st.slider("WebP quality", 60, 95, DEFAULT_QUALITY)
    render_size = st.selectbox("Render size", API_IMAGE_SIZE_OPTIONS, index=0)
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
    placeholder="things to do in Vail in April\nbest seafood restaurant in Boston\nwhat county is Vail\nlift ticket price Vail",
)

c1, c2 = st.columns(2)
run = c1.button("Generate")
if c2.button("Clear"):
    st.session_state.clear()
    st.experimental_rerun()

# =========================
# Run
# =========================
def do_one(api_key: str, keyword: str, site_key: str, render_size: str, quality: int,
           serpapi_key: str) -> Tuple[str, bytes, Optional[str], List[str], List[str]]:
    """
    Returns: (final_prompt, webp_bytes, facts_hint, ref_titles, ref_cues)
    """
    site_vibe = SITE_PRESETS.get(site_key, SITE_PRESETS[DEFAULT_SITE])

    # Season directive from keyword + site
    season_hint = season_hint_for_site(site_key, keyword)

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
    base_prompt = plan_prompt(api_key, site_vibe, keyword, site_key, facts_hint, ref_cues, season_hint)

    # Critic (single pass)
    final_prompt = critique_and_refine(api_key, keyword, base_prompt)

    # Image
    png = generate_image_bytes(api_key, final_prompt, render_size)
    img = Image.open(io.BytesIO(png))
    return final_prompt, to_webp_bytes(img, OUT_W, OUT_H, quality), facts_hint, ref_titles, ref_cues

if run:
    if not openai_key:
        st.warning("Please enter your OpenAI API key.")
        st.stop()

    kws: List[str] = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
    if not kws:
        st.warning("Please paste at least one keyword.")
        st.stop()

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
                openai_key, kw, site, render_size, quality, serpapi_key
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
                           file_name=f"{slugify(site)}_images.zip",
                           mime="application/zip", key="zip")

        st.markdown("### Previews")
        cols = st.columns(3)
        for idx, (fname, data_bytes) in enumerate(thumbs):
            with cols[idx % 3]:
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
