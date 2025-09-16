# ImageForge Autopilot v1.0.1
# Bulk, plug-and-play blog images with per-image downloads + ZIP export.
# Now with indoor/outdoor intent steering & strong negatives.
# Requirements: streamlit, requests, pillow

import base64
import io
import random
import re
import zipfile
from typing import List, Tuple

import requests
from PIL import Image
import streamlit as st

APP_NAME = "ImageForge Autopilot"
APP_VERSION = "v1.0.1"

st.set_page_config(page_title=f"{APP_NAME} — {APP_VERSION}", layout="wide")

# Optional: put APP_PASSWORD or OPENAI_API_KEY in Streamlit Secrets
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")
DEFAULT_SITE = "vailvacay.com"

SITE_PRESETS = {
    "vailvacay.com":  "Colorado alpine resort & village; paths, rivers, gondola stations, cozy lodges; photorealistic; no text.",
    "bostonvacay.com": "New England city; Beacon Hill brownstones, harborwalk, Charles River, stations; photorealistic; no text.",
    "bangkokvacay.com":"Southeast Asian metropolis; temples, canals, night markets, BTS/MRT; photorealistic; no text.",
    "ipetzo.com":      "Pet lifestyle; dogs/cats with owners in parks or cozy homes; neutral interiors; photorealistic; no brands; no text.",
    "1-800deals.com":  "Retail/e-commerce visuals; parcels, unboxing, generic products; clean backgrounds; photorealistic; no brands; no text.",
}

API_IMAGE_SIZE = "1536x1024"     # Supported: 1024x1024, 1024x1536, 1536x1024
OUTPUT_W, OUTPUT_H = 1200, 675   # Blog target
DEFAULT_QUALITY = 82

# ---------------- Utils ----------------
def slugify(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[’'`]", "", t)
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t

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

# ------------- Prompt planner -------------
def site_negatives(site: str) -> str:
    common = "No text/typography; avoid readable signs/logos; privacy-respecting; non-explicit."
    if site == "1-800deals.com":
        return common + " Packaging must be generic with blank labels; avoid trademarks/SKUs/barcodes."
    if site == "ipetzo.com":
        return common + " No unsafe handling or harmful gear."
    if site == "bostonvacay.com":
        return common + " Use generic wayfinding; no airline/transit branding."
    if site == "vailvacay.com":
        return common + " Avoid generic condo rows unless the keyword is lodging."
    if site == "bangkokvacay.com":
        return common + " Nightlife scenes: exterior/cabaret entrance only; no lewd or explicit content."
    return common

def env_from_keyword(keyword: str) -> str:
    k = keyword.lower()
    if any(w in k for w in ["indoor", "inside", "rainy day", "rainy-day", "bad weather", "winter activities indoors"]):
        return "indoor"
    if any(w in k for w in ["outdoor", "outside", "hike", "trail", "scenic", "vista"]):
        return "outdoor"
    return "auto"

def season_hint(keyword: str) -> str:
    k = keyword.lower()
    if any(w in k for w in ["winter", "snow", "dec", "jan", "feb", "christmas"]):
        return "winter atmosphere if relevant"
    if any(w in k for w in ["fall", "autumn", "sept", "oct", "nov", "foliage"]):
        return "autumn foliage if relevant"
    if any(w in k for w in ["spring", "apr", "may"]):
        return "spring greens if relevant"
    if any(w in k for w in ["summer", "jun", "jul", "aug", "patio"]):
        return "summer warmth if relevant"
    return "seasonally appropriate light"

def subject_cues(keyword: str, env: str) -> Tuple[List[str], List[str]]:
    """Returns (soft_cues, strong_refine_cues)."""
    k = keyword.lower()
    cues, strong = [], []

    # Indoor / Outdoor steering (first so it dominates)
    if env == "indoor":
        cues.append("indoor setting with visible ceiling and walls, ambient lighting, windows optional")
        strong.append("clear interior architecture (ceiling + walls) and activity context; exclude outdoor vistas")
    elif env == "outdoor":
        cues.append("outdoor setting with natural light and terrain features")

    # Family / baby
    if any(w in k for w in ["baby", "infant", "kid", "family", "with kids"]):
        if env == "indoor":
            cues.append("family-friendly indoor activity such as play corner, climbing wall, skating, or rec room")
        else:
            cues.append("parent with stroller or front baby carrier on a safe path or play area")
        strong.append("parent clearly pushing a stroller or wearing a front baby carrier in the foreground")

    # Transport / venues / categories
    if any(w in k for w in ["gondola", "ski lift", "lift tickets"]):
        cues.append("gondola cabin or station visibly in frame")
        strong.append("a gondola cabin prominently in the scene near the subject")

    if any(w in k for w in ["restaurant", "burger", "sushi", "dining", "eat", "cafe", "breakfast", "brunch", "patio"]):
        cues.append("dining context with a plated dish or table setting (no logos, no readable menus)")
        strong.append("a plated dish on a table in a dining setting, no logos")

    if any(w in k for w in ["ferry", "boat tour", "harbor cruise"]):
        cues.append("ferry terminal or boarding ramp with vessel visible (no readable vessel names)")
        strong.append("a ferry or boarding ramp clearly visible near travelers")

    if any(w in k for w in ["train", "amtrak", "station"]):
        cues.append("train platform or station concourse context (generic wayfinding)")
        strong.append("a train or platform edge clearly visible with travelers")

    if any(w in k for w in ["flight", "airport", "terminal", "customs", "bkk", "suvarnabhumi", "don mueang"]):
        cues.append("airport terminal gates or wing-view context (no airline logos)")
        strong.append("airport gate area with aircraft outside the windows, no logos")

    if any(w in k for w in ["ping pong show", "soapy", "nuru", "ladyboy", "red light", "cabaret", "freelancer", "hooker", "short time"]):
        cues.append("nightlife exterior streetscape or cabaret entrance only; safe, non-explicit")
        strong.append("cabaret entrance exterior with lights and crowd silhouettes; non-explicit")

    if any(w in k for w in ["passport photo", "id photo", "visa photo"]):
        cues.append("small photo studio setup with neutral backdrop, softbox lights, camera on tripod")
        strong.append("photo studio backdrop and softbox lights clearly visible")

    if any(w in k for w in ["vape", "relx"]):
        cues.append("generic electronics kiosk with unbranded vape-style devices")
        strong.append("unbranded vape devices on a kiosk shelf; no logos")

    return cues, strong

def build_prompt(site: str, keyword: str, variation: str = "") -> Tuple[str, List[str]]:
    base = SITE_PRESETS.get(site, SITE_PRESETS[DEFAULT_SITE])
    negs = site_negatives(site)
    env = env_from_keyword(keyword)
    cues, strong = subject_cues(keyword, env)

    # Environment-specific negatives to prevent model drift
    env_negs = ""
    if env == "indoor":
        env_negs = " Exclude rivers, gondolas, ski slopes, and wide outdoor landscape vistas."
    elif env == "outdoor":
        env_negs = " Avoid obvious indoor scenes unless essential."

    composition = (
        "balanced composition, natural light, editorial stock-photo feel; "
        "clear foreground subject and layered background; landscape orientation."
    )
    season = season_hint(keyword)

    chips = [composition, season, negs + env_negs]
    if variation:
        chips.append(variation)

    # Primary brief (short, decisive)
    core = f"Create a photorealistic travel blog image for: '{keyword}'. "
    if env == "indoor":
        core += "Use an indoor activity scene only. "
    elif env == "outdoor":
        core += "Use an outdoor scene. "
    core += f"Site vibe: {base} "

    if cues:
        core += "Include clear visual cues: " + "; ".join(cues) + ". "

    prompt = core + " ".join(chips) + " No typography anywhere."
    return prompt, strong

def refine_prompt(original_prompt: str, strong_cues: List[str]) -> str:
    if not strong_cues:
        return original_prompt + " Make the subject unmistakable in frame."
    return original_prompt + " Make the subject unmistakable in frame. Emphasize: " + "; ".join(strong_cues) + "."

def dalle_image_bytes(prompt: str, size: str, api_key: str) -> bytes:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "gpt-image-1", "prompt": prompt, "size": size}
    r = requests.post("https://api.openai.com/v1/images/generations",
                      headers=headers, json=payload, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text}")
    data0 = r.json()["data"][0]
    if "b64_json" in data0:
        return base64.b64decode(data0["b64_json"])
    if "url" in data0:
        resp = requests.get(data0["url"], timeout=180)
        resp.raise_for_status()
        return resp.content
    raise RuntimeError("Images API returned neither b64_json nor url")

# ------------- UI -------------
st.title(f"{APP_NAME} — {APP_VERSION}")
st.caption("Paste keywords (one per line), choose a site, click Generate. The app plans a brief per keyword, "
           "adds subject cues, forces indoor/outdoor when requested, and outputs 1200×675 WebP images with per-image downloads + ZIP.")

if APP_PASSWORD:
    gate = st.text_input("Team password", type="password")
    if gate != APP_PASSWORD:
        st.stop()

with st.sidebar:
    site = st.selectbox("Site", list(SITE_PRESETS.keys()),
                        index=list(SITE_PRESETS.keys()).index(DEFAULT_SITE))
    quality = st.slider("WebP quality", 60, 95, DEFAULT_QUALITY)
    size = st.selectbox("Render size", ["1536x1024", "1024x1024", "1024x1536"], index=0)

    with st.expander("Advanced (optional)"):
        variations = [
            "",  # none
            "golden hour sunlight",
            "bluebird sky",
            "light snow flurries",
            "after-rain reflections and neon glow",
            "closer, human-forward framing",
            "wider establishing vista",
        ]
        variation_choice = st.selectbox("Global vibe tweak", variations, index=0)

openai_key = st.text_input(
    "OpenAI API key",
    type="password",
    value=st.secrets.get("OPENAI_API_KEY", ""),
    help="You can also set OPENAI_API_KEY in Streamlit Secrets."
)

keywords_text = st.text_area(
    "Keywords (one per line)",
    height=260,
    placeholder=(
        "Indoor activities Vail\n"
        "BTS to Chinatown Bangkok\n"
        "Boston to Bar Harbor ferry\n"
        "Best restaurants in Vail"
    ),
)

col1, col2 = st.columns([1,1])
run = col1.button("Generate")
if col2.button("Clear"):
    st.session_state.clear()
    st.experimental_rerun()

# ------------- Run -------------
if run:
    if not openai_key:
        st.warning("Please enter your OpenAI API key.")
        st.stop()

    keywords: List[str] = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
    if not keywords:
        st.warning("Please paste at least one keyword.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()
    zip_buf = io.BytesIO()
    zipf = zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED)
    thumbs: List[Tuple[str, bytes]] = []
    successes = 0

    for i, kw in enumerate(keywords, start=1):
        slug = slugify(kw)
        try:
            status.text(f"Generating {i}/{len(keywords)}: {kw}")

            prompt, strong_cues = build_prompt(site, kw, variation_choice)
            png_bytes = dalle_image_bytes(prompt, size, openai_key)
            img = Image.open(io.BytesIO(png_bytes))

            # Auto-refine once if there are strong cues (helps enforce indoor/outdoor + subject)
            if strong_cues:
                try:
                    refined = refine_prompt(prompt, strong_cues)
                    png_bytes2 = dalle_image_bytes(refined, size, openai_key)
                    img = Image.open(io.BytesIO(png_bytes2))
                except Exception:
                    pass

            webp = to_webp_bytes(img, OUTPUT_W, OUTPUT_H, quality)
            zipf.writestr(f"{slug}.webp", webp)
            thumbs.append((f"{slug}.webp", webp))
            successes += 1

        except Exception as e:
            st.error(f"{kw}: {e}")

        progress.progress(i / len(keywords))

    zipf.close()
    zip_buf.seek(0)

    if successes == 0:
        st.error("No images were generated. Check your API key and try again with 1–2 keywords.")
    else:
        st.success("Done! Download your images below.")
        st.download_button(
            "⬇️ Download ZIP",
            data=zip_buf,
            file_name=f"{slugify(site)}_images.zip",
            mime="application/zip",
            key="zip_dl"
        )

        st.markdown("### Previews")
        cols = st.columns(3)
        for idx, (fname, data_bytes) in enumerate(thumbs):
            with cols[idx % 3]:
                st.image(data_bytes, caption=fname, use_container_width=True)
                st.download_button(
                    "Download",
                    data=data_bytes,
                    file_name=fname,
                    mime="image/webp",
                    key=f"dl_{idx}"
                )
