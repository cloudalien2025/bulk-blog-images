import base64
import os, io, re, time, zipfile, requests
from typing import List
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Bulk Blog Image Generator", layout="wide")

# Optional simple team password gate. Leave "" to disable or set via Secrets.
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")

SITE_PROFILES = {
    "vailvacay.com":  "Photorealistic alpine resort & village scenes in the Colorado Rockies; ski terrain; evergreen forests; cozy lodges; no text.",
    "bangkokvacay.com":"Photorealistic Southeast Asian city scenes; temples, canals, street food markets, tuk-tuks; golden-hour light; no text.",
    "bostonvacay.com": "Photorealistic New England city imagery; brick townhouses, harbor, fall foliage, cobblestones; no text.",
    "ipetzo.com":      "Photorealistic pet lifestyle images; dogs/cats with owners outdoors/indoors; neutral interiors/parks; no brands; no text.",
    "1-800deals.com":  "Photorealistic retail/ecommerce visuals; shopping scenes, parcels, generic products; clean backgrounds; no brands; no text.",
}
DEFAULT_SITE = "vailvacay.com"
API_IMAGE_SIZE = "1536x1024"      # Supported: 1024x1024, 1024x1536, 1536x1024, or "auto"
OUTPUT_W, OUTPUT_H = 1200, 675    # final blog size
DEFAULT_QUALITY = 82              # WebP quality (70–82 is good)

# ---------- helpers ----------
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

def save_webp_bytes(img: Image.Image, w: int, h: int, quality: int) -> bytes:
    img = img.convert("RGB")
    img = crop_to_aspect(img, w, h).resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=quality, method=6)
    return buf.getvalue()

def build_prompt(site: str, keyword: str) -> str:
    base = SITE_PROFILES.get(site, SITE_PROFILES[DEFAULT_SITE])
    k = keyword.lower(); style_hints = []
    if any(x in k for x in ["how far","get to","how to get","directions","from avon","from colorado springs"]):
        style_hints.append("scenic travel approach or roadway vantage; safe roadside view; vehicles optional")
    if any(x in k for x in ["back bowls","snow","open","ski"]):
        style_hints.append("wide landscape vista that fits the topic; seasonally appropriate terrain")
    if any(x in k for x in ["where to stay","hotel review","hotel"]):
        style_hints.append("inviting lodging exterior or interior; golden-hour or dusk; warm light")
    if "cigar" in k:
        style_hints.append("cozy upscale lounge interior; leather seating; warm ambient light; no branding")
    if any(x in k for x in ["don't ski","don’t ski","non-skiers","baby-friendly","where to hike"]):
        style_hints.append("walkable areas or easy trails; friendly human presence; relaxed mood")
    if any(x in k for x in ["what county","what river","what mountain range","when was","when did","how corporate"]):
        style_hints.append("documentary/editorial feel emphasizing place over people")
    if "celebrities" in k:
        style_hints.append("privacy-respecting generic high-end neighborhood; no identifiable faces")
    if "how to dress" in k:
        style_hints.append("tasteful street-style outfit details; no visible logos")
    if any(x in k for x in ["beaver creek","compared to vail"]):
        style_hints.append("neutral comparative vibe; tasteful resort scenery")
    style = ", ".join(style_hints) if style_hints else "scene appropriate to the topic"
    return (f"{base} Balanced composition; natural light; editorial stock-photo feel. "
            f"Create an image for the topic: '{keyword}'. Landscape orientation. No words or typography anywhere. "
            f"Scene intent: {style}.")

def dalle_generate_image_bytes(prompt: str, size: str, api_key: str) -> bytes:
    """
    Call Images API. Prefer base64 if returned; otherwise fall back to URL download.
    Returns raw PNG bytes.
    """
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

# ---------- UI ----------
st.title("Bulk Blog Image Generator (1200×675 WebP)")
st.caption("Paste keywords (one per line). Generates DALL·E images, crops to 1200×675, then bundles a ZIP.")

# Optional password gate
if APP_PASSWORD:
    pwd = st.text_input("Team password", type="password")
    if pwd != APP_PASSWORD:
        st.stop()

with st.sidebar:
    site = st.selectbox("Site style", list(SITE_PROFILES.keys()),
                        index=list(SITE_PROFILES.keys()).index(DEFAULT_SITE))
    quality = st.slider("WebP quality", 60, 95, DEFAULT_QUALITY)
    size = st.selectbox("DALL·E render size", ["1536x1024","1024x1536","1024x1024"], index=0)

keywords_text = st.text_area(
    "Keywords (one per line)", height=240,
    placeholder="Where to Stay in Vail\nWhat County is Vail Colorado in\nWhen Do Vail Back Bowls Open"
)
user_api_key = st.text_input("Enter your OpenAI API key", type="password",
                             help="Each user can use their own key.")

col1, col2 = st.columns([1,1])
run_btn = col1.button("Generate Images")
clear_btn = col2.button("Clear")
if clear_btn:
    st.session_state.clear()
    st.experimental_rerun()

if run_btn:
    if not user_api_key:
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
    thumbs = []
    successes = 0

    for i, kw in enumerate(keywords, start=1):
        try:
            slug = slugify(kw)
            status.text(f"Generating {i}/{len(keywords)}: {kw}")

            # Generate → bytes (PNG), then convert to 1200×675 WebP
            png_bytes = dalle_generate_image_bytes(build_prompt(site, kw), size, user_api_key)
            img = Image.open(io.BytesIO(png_bytes))
            webp_bytes = save_webp_bytes(img, OUTPUT_W, OUTPUT_H, quality)

            zipf.writestr(f"{slug}.webp", webp_bytes)
            thumbs.append((f"{slug}.webp", webp_bytes))
            successes += 1
        except Exception as e:
            st.error(f"{kw}: {e}")
        progress.progress(i / len(keywords))

    zipf.close()
    zip_buf.seek(0)

    if successes == 0:
        st.error("No images were generated. Check your API key or try a single keyword.")
    else:
        st.success("Done! Download your images below.")
        st.download_button("⬇️ Download ZIP", data=zip_buf,
                           file_name=f"{slugify(site)}_images.zip", mime="application/zip")

        st.markdown("### Preview & individual downloads")
        cols = st.columns(3)
        for idx, (fname, data_bytes) in enumerate(thumbs):
            with cols[idx % 3]:
                st.image(data_bytes, caption=fname, use_container_width=True)
                st.download_button("Download", data=data_bytes, file_name=fname, mime="image/webp")
