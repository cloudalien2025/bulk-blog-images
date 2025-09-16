# ImageForge Autopilot v2.1.0 — Planner + Critic
# Rule-light bulk image generation for blogs.
# AI plans & self-refines prompts; generates 1200×675 WebP images.
# Per-image downloads + ZIP export.
#
# Requirements: streamlit, requests, pillow

import io
import re
import json
import base64
import zipfile
from typing import List, Tuple

import requests
from PIL import Image
import streamlit as st

# =========================
# App identity & config
# =========================
APP_NAME = "ImageForge Autopilot"
APP_VERSION = "v2.1.0"

st.set_page_config(page_title=f"{APP_NAME} — {APP_VERSION}", layout="wide")

# Optional: password gate & default site (can be set in Secrets)
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

API_IMAGE_SIZE = "1536x1024"   # Supported: 1024x1024, 1024x1536, 1536x1024, or "auto"
OUT_W, OUT_H = 1200, 675       # Blog target
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
    """Parse JSON that might be wrapped in code fences."""
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
# Planner + Critic
# =========================

# Base planner instructions (site-agnostic + pattern nudges)
PLANNER_BASE = """
You are a senior creative director writing photorealistic image briefs
for a travel/consumer blog. Given a keyword and a site vibe, craft ONE concise prompt
(1–2 sentences) for a DALLE-like model. Infer season, indoor/outdoor, and obvious visual
cues so the topic is unmistakable. Use editorial stock-photo style.

Hard constraints (always include as plain text in the prompt):
- No text/typography overlays, no readable logos or brand marks, privacy-respecting, non-explicit.
- Balanced composition, natural light, landscape orientation.

Pattern nudges for question-style keywords (examples, not outputs):
- "what county is …": prefer a tabletop regional map with a pin near the destination, a hand placing the pin; OR an exterior of the county courthouse with the seal out of focus. Avoid readable text.
- "what river runs through …": make the river the hero (close/mid view of the water through town), with the place context secondary.
- "what mountain range is … in": emphasize defining ridgelines/peaks; town minimal.
- "how far is … from …": travel-planning vibe (map with two pinned points or dashboard/GPS scene); avoid readable text.
"""

# Tiny, optional site-specific nudges
SITE_NUDGES = {
    "vailvacay.com": [
        "Snow topics: if ski/back bowls/lift tickets appear, winter conditions are natural; think snowy slopes and lift infrastructure.",
    ],
    "bostonvacay.com": [
        "Ferry/Bar Harbor trips: boarding ramp or terminal scene with vessel in frame; signage generic/debranded.",
        "Wardrobe seasons: show layered outfits appropriate to the month; no visible logos.",
        "Birthday freebies: cozy café/pastry counter, treat on a plate; point-of-sale blurred; no readable signage.",
    ],
    "bangkokvacay.com": [
        "BTS/Chinatown: BTS platform with arriving train; wayfinding shapes/colors suggestive but unreadable; Chinatown lanterns hinted.",
        "Passport/visa photo: small studio corner with backdrop + softbox + tripod; no brands.",
        "Nightlife terms: exterior cabaret entrance / neon street scene; silhouettes only; non-explicit.",
    ],
    "ipetzo.com": [
        "Dog-friendly: owner with dog on patio or lobby seating; water bowl; no hotel branding.",
        "Indoor pet activities: living-room play scene, toys/bed visible; natural light; no brand marks.",
    ],
    "1-800deals.com": [
        "Unboxing/deals/coupons: generic parcels and product tableaus; device screens/descriptive labels out-of-focus; blank shipping labels; no trademarks.",
    ],
}

def build_planner_system(site_key: str) -> str:
    nudges = SITE_NUDGES.get(site_key, [])
    site_block = ""
    if nudges:
        site_block = "\nSite-specific nudges:\n- " + "\n- ".join(nudges) + "\n"
    return (
        PLANNER_BASE
        + site_block
        + '\nOutput JSON ONLY:\n{"prompt": "<final one- or two-sentence prompt>"}'
    )

CRITIC_SYSTEM = """
You are a prompt critic. Given a keyword and a proposed image prompt,
decide if the prompt obviously covers the keyword. If missing, revise succinctly.
Keep style constraints intact (no text/logos, non-explicit; balanced composition; landscape).
Output JSON ONLY:
- If OK: {"action":"ok"}
- If needs fix: {"action":"refine","prompt":"<better prompt>"}
"""

def plan_prompt(api_key: str, site_vibe: str, keyword: str, global_vibe: str, site_key: str) -> str:
    planner_system = build_planner_system(site_key)
    user = f"""Site vibe: {site_vibe}
Keyword: {keyword}
Global vibe tag (optional): {global_vibe}

Write a single, photorealistic travel-blog prompt that:
- Clearly signals the topic with obvious visual cues
- Chooses indoor vs outdoor and season naturally from the keyword
- Stays safe and brand-neutral
- Is concise (max ~2 sentences)"""
    content = chat_completion(api_key, [
        {"role":"system","content":planner_system},
        {"role":"user","content":user}
    ])
    data = extract_json(content)
    if not data or "prompt" not in data:
        # Fall back to raw text if JSON parse fails
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
st.caption("Paste keywords (one per line). The AI plans and self-refines prompts, then generates 1200×675 WebP images. Rule-light, brand-safe visuals.")

if APP_PASSWORD:
    pw = st.text_input("Team password", type="password")
    if pw != APP_PASSWORD:
        st.stop()

with st.sidebar:
    site = st.selectbox("Site", list(SITE_PRESETS.keys()),
                        index=list(SITE_PRESETS.keys()).index(DEFAULT_SITE))
    quality = st.slider("WebP quality", 60, 95, DEFAULT_QUALITY)
    size = st.selectbox("Render size", ["1536x1024", "1024x1024", "1024x1536"], index=0)
    with st.expander("Advanced (optional)"):
        global_vibe = st.selectbox(
            "Global vibe tag (optional)",
            ["", "golden hour", "bluebird sky", "light snow", "after-rain glow",
             "human-forward framing", "wider vista", "cozy indoor"],
            index=0
        )

openai_key = st.text_input(
    "OpenAI API key",
    type="password",
    value=st.secrets.get("OPENAI_API_KEY", ""),
    help="Or set OPENAI_API_KEY in Streamlit Secrets."
)

keywords_text = st.text_area(
    "Keywords (one per line)",
    height=280,
    placeholder="Things to Do in Vail with a Baby\nIndoor activities Vail\nBoston to Bar Harbor ferry\nBTS to Chinatown Bangkok\nWhat county is Vail",
)

c1, c2 = st.columns(2)
run = c1.button("Generate")
if c2.button("Clear"):
    st.session_state.clear()
    st.experimental_rerun()

# =========================
# Run
# =========================
def do_one(api_key: str, keyword: str, site_key: str, global_vibe: str, render_size: str, quality: int) -> Tuple[str, bytes]:
    site_vibe = SITE_PRESETS.get(site_key, SITE_PRESETS[DEFAULT_SITE])

    # 1) Planner
    base_prompt = plan_prompt(api_key, site_vibe, keyword, global_vibe or "auto", site_key=site_key)

    # 2) Critic (single-pass refine if needed)
    final_prompt = critique_and_refine(api_key, keyword, base_prompt)

    # 3) Image
    png = generate_image_bytes(api_key, final_prompt, render_size)
    img = Image.open(io.BytesIO(png))
    return final_prompt, to_webp_bytes(img, OUT_W, OUT_H, quality)

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
    prompts_used: List[Tuple[str, str]] = []  # (fname, prompt)
    zip_buf = io.BytesIO()
    zipf = zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED)

    for i, kw in enumerate(kws, start=1):
        fname = f"{slugify(kw)}.webp"
        try:
            status.text(f"Generating {i}/{len(kws)}: {kw}")
            prompt_used, webp = do_one(openai_key, kw, site, global_vibe, size, quality)
            zipf.writestr(fname, webp)
            thumbs.append((fname, webp))
            prompts_used.append((fname, prompt_used))
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

        with st.expander("Prompts used (for QA)"):
            for fname, p in prompts_used:
                st.markdown(f"**{fname}**")
                st.code(p, language="text")
