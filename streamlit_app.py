# streamlit_app.py
import base64
import os, io, re, time, zipfile, requests, random
from typing import List
from PIL import Image
import streamlit as st

# -------------------- App setup --------------------
st.set_page_config(page_title="Bulk Blog Image Generator", layout="wide")

# Optional simple team password (set in Secrets -> APP_PASSWORD or hardcode here)
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")  # "" disables the gate

SITE_PROFILES = {
    "vailvacay.com":  "Photorealistic alpine resort & village scenes in the Colorado Rockies; ski terrain; evergreen forests; cozy lodges; no text.",
    "bangkokvacay.com":"Photorealistic Southeast Asian city scenes; temples, canals, street food markets, tuk-tuks; golden-hour light; no text.",
    "bostonvacay.com": "Photorealistic New England city imagery; Beacon Hill brick townhouses, harbor, Charles River, fall foliage; no text.",
    "ipetzo.com":      "Photorealistic pet lifestyle images; dogs/cats with owners outdoors/indoors; neutral interiors/parks; no brands; no text.",
    "1-800deals.com":  "Photorealistic retail/ecommerce visuals; shopping scenes, parcels, generic products; clean backgrounds; no brands; no text.",
}
DEFAULT_SITE = "vailvacay.com"

API_IMAGE_SIZE = "1536x1024"      # Supported: 1024x1024, 1024x1536, 1536x1024, or "auto"
OUTPUT_W, OUTPUT_H = 1200, 675    # Final blog size
DEFAULT_QUALITY = 82              # WebP quality (70–82 is good)

# -------------------- Helpers --------------------
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

# ====== UNIVERSAL SMART PROMPT PLANNER ======
def detect_season(keyword: str) -> str:
    k = keyword.lower()
    if any(w in k for w in ["winter", "snow", "dec", "jan", "feb"]): return "winter"
    if any(w in k for w in ["fall", "autumn", "sept", "oct", "nov", "foliage"]): return "fall"
    if any(w in k for w in ["spring", "apr", "may", "wildflower"]): return "spring"
    if any(w in k for w in ["summer", "jun", "jul", "aug", "lake", "river", "hike", "paddle", "raft"]): return "summer"
    return "auto"

def season_descriptors(season: str) -> str:
    return {
        "winter": "fresh snow, crisp air, low winter sun; conifers frosted",
        "summer": "lush greens, clear sky, warm light; water activity possible",
        "fall":   "golden foliage, warm sidelight, long shadows",
        "spring": "snowmelt, budding trees, bright fresh greens",
        "auto":   "seasonally-appropriate light and foliage"
    }[season]

def composition_hint() -> str:
    return ("Composition: strong foreground interest, clear midground subject, layered background; "
            "eye-level camera, 24–35mm FoV, gentle leading lines from a path/road/water.")

def negatives(site: str) -> str:
    common = "No typography overlay; avoid readable signs or brand logos."
    if site == "1-800deals.com":
        return common + " Packaging must be generic with blank labels; avoid trademarks/SKUs/barcodes."
    if site == "ipetzo.com":
        return common + " No harmful gear or unsafe handling."
    return common

def classify(keyword: str) -> set:
    k = keyword.lower()
    tags = set()
    if "things to do" in k: tags.add("todo")
    if any(w in k for w in ["how far", "distance", "drive", "get to", "directions"]): tags.add("drive")
    if any(w in k for w in ["where to stay", "hotel", "lodging", "resort"]): tags.add("hotel")
    if any(w in k for w in ["hike", "trail"]): tags.add("hike")
    if any(w in k for w in ["baby", "family", "stroller", "non-skiers", "don't ski", "don’t ski"]): tags.add("family")
    if "cigar" in k: tags.add("cigar")
    if any(w in k for w in ["compare", "compared to"]): tags.add("compare")
    if "between" in k and "vail" in k and "denver" in k: tags.add("between_vail_denver")
    # Pet-site specific intents
    if any(w in k for w in ["groom", "grooming", "bath"]): tags.add("grooming")
    if any(w in k for w in ["train", "obedience", "recall"]): tags.add("training")
    if any(w in k for w in ["vet", "clinic", "checkup"]): tags.add("vet")
    if any(w in k for w in ["travel", "car", "hotel", "carrier", "flight"]): tags.add("pet_travel")
    if any(w in k for w in ["cat", "kitten"]): tags.add("cat")
    if any(w in k for w in ["dog", "puppy"]): tags.add("dog")
    # Retail intents
    if any(w in k for w in ["deals", "coupon", "sale", "shopping", "checkout", "unboxing", "parcel"]):
        tags.add("retail")
    return tags

def choose_one(options): return random.choice(options)

def geo_enrichment(site: str, keyword: str, season: str, tags: set) -> str:
    s = site.lower()

    # ----- Vail Vacay -----
    if s == "vailvacay.com":
        if "between_vail_denver" in tags:
            winter = [
                "Loveland Pass overlook with winding US-6 switchbacks and Continental Divide peaks",
                "Georgetown Loop Railroad crossing a steel trestle over a snowy pine valley",
                "Idaho Springs historic main street with light snow and steam from hot springs area",
                "Dillon Reservoir shoreline with frozen lake activities and Tenmile Range backdrop",
            ]
            summer = [
                "Red Rocks Park approach with sandstone fins and foothills at golden hour",
                "Dillon Reservoir boardwalk with kayaks and the Tenmile Range behind",
                "Georgetown Loop Railroad on a trestle over pine valley in summer greens",
                "Idaho Springs main street with Victorian brick storefronts and foothills",
            ]
            return choose_one(winter if season == "winter" else summer)
        if "drive" in tags:
            return "scenic I-70 mountain approach framed by evergreens and peaks; safe roadside vantage"
        if "hotel" in tags:
            return "inviting alpine-style hotel exterior at dusk with warm window glow and mountains behind"
        if "hike" in tags or "family" in tags:
            return "easy riverside path or lakeside trail with families; aspens and peaks around"
        if "cigar" in tags:
            return "cozy upscale lounge interior with leather chairs and warm ambient light"
        return "place-relevant alpine hero scene showing mountains, forest, and village context"

    # ----- Bangkok Vacay -----
    if s == "bangkokvacay.com":
        if "todo" in tags:
            return choose_one([
                "floating market with boats of fruit and flowers on a canal",
                "Wat Arun riverside view at golden hour with a long-tail boat passing",
                "Chinatown night street food scene with steam and neon ambience (no readable signs)",
                "Tuk-tuk on a lively old-town street near temples and palms",
            ])
        if "drive" in tags:
            return "riverfront arterial with skyline glimpse and long-tail boat; warm sunset"
        return choose_one([
            "temple courtyard with ornate roofs and palms",
            "canal scene with wooden houses and boats",
            "night market with colorful stalls and crowds (blurred, no readable text)"
        ])

    # ----- Boston Vacay -----
    if s == "bostonvacay.com":
        if "todo" in tags:
            return choose_one([
                "Beacon Hill brownstones with gas lamps and brick sidewalks",
                "Boston Public Garden footbridge with swan boats",
                "Charles River Esplanade with runners and sailboats on the water",
                "Harborwalk with skyline across the harbor at golden hour",
            ])
        if "drive" in tags:
            return "riverside vista along Storrow Drive with Charles River and skyline beyond"
        if "hotel" in tags:
            return "classic Back Bay hotel facade at golden hour with city skyline hints"
        if "hike" in tags or "family" in tags:
            return "Emerald Necklace park path with families and strollers under leafy trees"
        return choose_one([
            "cobblestone lane with historic brick facades",
            "harbor marina with boats and skyline",
            "leafy neighborhood street with historical character"
        ])

    # ----- iPetzo (pets) -----
    if s == "ipetzo.com":
        if "grooming" in tags:
            return "bright grooming salon scene with a dog being trimmed on a table; tidy tools; no branding"
        if "training" in tags:
            return "obedience training in a park; handler rewarding a dog with treats; correct leash handling"
        if "vet" in tags:
            return "modern vet clinic exam room; vet gently checking a pet; clean, unbranded space"
        if "pet_travel" in tags:
            return "pet-friendly hotel room or car with a safely harnessed dog; travel gear without logos"
        if "cat" in tags:
            return "sunny living-room window light with a cat on a couch or cat tree; tasteful minimal decor"
        # default dogs/pets lifestyle
        return "owners walking a dog on a leafy path or city park; friendly mood; neutral collars with no logos"

    # ----- 1-800deals (retail) -----
    if s == "1-800deals.com":
        if "retail" in tags or "todo" in tags:
            return choose_one([
                "unboxing scene on a clean table with plain cardboard box and tissue paper (no logos or text)",
                "shopping cart close-up in a bright generic store aisle with abstract products (no logos)",
                "parcel stack at a doorstep with neutral labels and tape (no branding)",
                "hands at a generic checkout counter scanning a plain box (no readable screens/text)",
            ])
        return "generic ecommerce product lifestyle scene with plain packaging and soft studio light"

    # ----- Fallback for unknown sites -----
    return "place-appropriate hero scene that fits the topic and feels specific to the region or niche"

def build_prompt(site: str, keyword: str) -> str:
    base = SITE_PROFILES.get(site, SITE_PROFILES[DEFAULT_SITE])
    season = detect_season(keyword)
    season_words = season_descriptors(season)
    tags = classify(keyword)
    scene = geo_enrichment(site, keyword, season, tags)
    comp = composition_hint()
    neg = negatives(site)
    photo_salt = random.choice([
        "documentary travel photo, natural color",
        "editorial stock photo, soft contrast",
        "RAW-like realism, accurate white balance",
        "crisp air clarity, minimal haze"
    ])
    people_hint = "subtle human presence for scale, people small in frame" if "todo" in tags else "people optional"

    return (
        f"{base} {photo_salt}. {comp} {neg} "
        f"Season cues: {season_words}. {people_hint}. "
        f"Keyword: '{keyword}'. Depict specifically: {scene}. "
        f"Landscape orientation only. No words or typography anywhere."
    )
# ====== END UNIVERSAL SMART PROMPT PLANNER ======

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

# -------------------- UI --------------------
st.title("Bulk Blog Image Generator (1200×675 WebP)")
st.caption("Paste keywords (one per line). Generates DALL·E images, crops to 1200×675, and bundles a ZIP.")

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
    "Keywords (one per line)",
    height=240,
    placeholder="Where to Stay in Vail\nThings To Do Between Vail and Denver\nDog Grooming Tips\nBest Deals on Kitchen Gadgets"
)
user_api_key = st.text_input("Enter your OpenAI API key", type="password",
                             help="Each user can use their own key (recommended).")

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
        st.error("No images were generated. Check your API key and try again with 1–2 keywords.")
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
