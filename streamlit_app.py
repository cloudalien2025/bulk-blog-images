# ImageForge v1.4 — Real Photos + AI Render + Batch from Excel
# ------------------------------------------------------------
# Requirements: streamlit, requests, pillow, pandas, openpyxl
#   pip install streamlit requests pillow pandas openpyxl
#
# IMPORTANT: Enable these Google APIs if using Real Photos mode:
#   • Places API (or Places API (New))
#   • Street View Static API

from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
from PIL import Image
import streamlit as st
import pandas as pd

APP_NAME = "ImageForge v1.4"

OUTPUT_W, OUTPUT_H = 1200, 675
PINTEREST_W, PINTEREST_H = 1000, 1500

OPENAI_IMAGE_SIZES = ["1024x1024", "1024x576", "576x1024"]

# Read keys from Streamlit Secrets (if provided)
SECRETS = st.secrets.get("api_keys", {})

@dataclass
class Candidate:
    title: str
    source: str
    preview_bytes: bytes
    license_note: str

def slugify(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[’'`]", "", t)
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t

def crop_resize_to(img: Image.Image, w: int, h: int) -> Image.Image:
    t_ratio = w / h
    iw, ih = img.size
    if iw / ih > t_ratio:
        new_w = int(ih * t_ratio)
        x0 = (iw - new_w) // 2
        box = (x0, 0, x0 + new_w, ih)
    else:
        new_h = int(iw / t_ratio)
        y0 = (ih - new_h) // 2
        box = (0, y0, iw, y0 + new_h)
    return img.crop(box).resize((w, h), Image.LANCZOS)

def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGB", "RGBA"):
        return img.convert("RGB")
    return img.convert("RGB")

def to_webp_bytes(img_bytes: bytes, w: int, h: int, quality: int = 82) -> bytes:
    img = Image.open(io.BytesIO(img_bytes))
    img = ensure_rgb(img)
    img = crop_resize_to(img, w, h)
    out = io.BytesIO()
    img.save(out, format="WEBP", quality=quality, method=6)
    return out.getvalue()

def google_textsearch_place(q: str, gmaps_key: str) -> Optional[dict]:
    r = requests.get("https://maps.googleapis.com/maps/api/place/textsearch/json", params={
        "query": q,
        "key": gmaps_key
    }, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json() or {}
    if data.get("status") not in ("OK", "ZERO_RESULTS"):
        return None
    results = data.get("results") or []
    return results[0] if results else None

def google_place_details(place_id: str, gmaps_key: str) -> dict:
    r = requests.get("https://maps.googleapis.com/maps/api/place/details/json", params={
        "place_id": place_id,
        "fields": "name,geometry,photos",
        "key": gmaps_key
    }, timeout=30)
    return (r.json() or {}).get("result", {}) if r.status_code == 200 else {}

def google_photo_bytes(photo_ref: str, gmaps_key: str, max_w: int = 1600) -> Optional[bytes]:
    url = "https://maps.googleapis.com/maps/api/place/photo"
    r = requests.get(url, params={"photoreference": photo_ref, "maxwidth": max_w, "key": gmaps_key},
                     timeout=30, allow_redirects=False)
    loc = r.headers.get("Location")
    if loc:
        img = requests.get(loc, timeout=30)
        if img.status_code == 200:
            return img.content
    if r.status_code == 200 and r.content:
        return r.content
    return None

def streetview_bytes(lat: float, lng: float, gmaps_key: str, radius_m: int = 250,
                     size_w: int = 1024, size_h: int = 1024) -> Optional[bytes]:
    meta = requests.get(
        "https://maps.googleapis.com/maps/api/streetview/metadata",
        params={"location": f"{lat},{lng}", "radius": radius_m, "key": gmaps_key},
        timeout=20,
    ).json()
    if meta.get("status") not in ("OK", "ZERO_RESULTS"):
        return None
    if meta.get("status") == "ZERO_RESULTS":
        return None
    r = requests.get(
        "https://maps.googleapis.com/maps/api/streetview",
        params={"location": f"{lat},{lng}", "radius": radius_m, "key": gmaps_key,
                "size": f"{size_w}x{size_h}"},
        timeout=20,
    )
    if r.status_code == 200 and r.content:
        return r.content
    return None

def serpapi_images(q: str, serp_key: str, num: int = 6) -> List[Tuple[str, str]]:
    """
    Return list of (source_label, image_url).
    """
    out: List[Tuple[str, str]] = []
    try:
        r = requests.get("https://serpapi.com/search.json", params={
            "engine": "google",
            "q": q,
            "tbm": "isch",
            "api_key": serp_key,
            "ijn": 0,
        }, timeout=30)
        if r.status_code != 200:
            return out
        data = r.json() or {}
        for img in (data.get("images_results") or [])[:num]:
            link = img.get("original") or img.get("thumbnail")
            if not link:
                continue
            src = img.get("source") or "Google Images"
            out.append((src, link))
    except Exception:
        pass
    return out

SITE_PROFILES = {
    "VailVacay.com": dict(
        base_prompt="High-altitude Rocky Mountain ski resort; editorial travel blog style.",
        color_hint="wintry blues, snow whites, warm lodge interiors",
    ),
    "BangkokVacay.com": dict(
        base_prompt="Tropical Southeast Asian megacity; street-food, temples, markets, and skyline.",
        color_hint="warm ambers, neon lights, lush greens",
    ),
}
DEFAULT_SITE = "VailVacay.com"

def build_ai_prompt(keyword: str, site: str) -> str:
    base_cfg = SITE_PROFILES.get(site, SITE_PROFILES[DEFAULT_SITE])
    base = base_cfg["base_prompt"]
    color_hint = base_cfg["color_hint"]
    k = keyword.lower()
    style_hints = [color_hint]
    if any(x in k for x in ["winter", "snow", "ski", "powder", "gondola", "lift"]):
        style_hints.append("snowy alpine environment; ski resort ambiance")
    if any(x in k for x in ["summer", "hike", "trail", "bike"]):
        style_hints.append("summer mountain setting; green trees, bright light")
    if any(x in k for x in ["november", "october"]):
        style_hints.append("shoulder season feel; transitional foliage")
    style = ", ".join(style_hints) if style_hints else "scene appropriate to the topic"
    return (f"{base} Create a photorealistic landscape-orientation image for: '{keyword}'. "
            f"Balanced composition; natural light; editorial stock-photo feel; "
            f"no text or logos; no brand marks. Scene intent: {style}.")

# -----------------------------
# Real-photo candidate collector
# -----------------------------

def collect_real_photo_candidates(q: str,
                                  use_places_flag: bool,
                                  use_street_flag: bool,
                                  use_serp_flag: bool,
                                  sv_radius_m: int,
                                  gmaps_key: str,
                                  serp_key: Optional[str]) -> List[Candidate]:
    cands: List[Candidate] = []
    if not gmaps_key:
        return cands

    place = google_textsearch_place(q, gmaps_key)
    if not place:
        return cands

    details = google_place_details(place["place_id"], gmaps_key)
    title = details.get("name") or q
    loc = (details.get("geometry") or {}).get("location") or {}
    lat, lng = loc.get("lat"), loc.get("lng")

    # Google Places Photos
    if use_places_flag:
        for ph in (details.get("photos") or [])[:12]:
            ref = ph.get("photo_reference")
            if not ref:
                continue
            try:
                img_bytes = google_photo_bytes(ref, gmaps_key, max_w=1600)
                if img_bytes:
                    cands.append(Candidate(
                        title=f"Google Places Photo — {title}",
                        source="Google Maps contributor",
                        preview_bytes=img_bytes,
                        license_note="License: Refer to Google Places Photo terms",
                    ))
            except Exception:
                pass

    # Google Street View
    if use_street_flag and lat and lng:
        try:
            sv = streetview_bytes(lat, lng, gmaps_key, radius_m=sv_radius_m, size_w=1024, size_h=1024)
            if sv:
                cands.append(Candidate(
                    title=f"Google Street View — {title}",
                    source="Google Street View",
                    preview_bytes=sv,
                    license_note="License: Refer to Google Street View terms",
                ))
        except Exception:
            pass

    # SerpAPI (reference-only)
    if use_serp_flag and serp_key:
        refs = serpapi_images(q, serp_key, num=6)
        for src, url in refs:
            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 200 and r.content:
                    cands.append(Candidate(
                        title=f"SerpAPI (Google Images, reference) — {title}",
                        source=src,
                        preview_bytes=r.content,
                        license_note="License: Refer to original site terms; for reference only.",
                    ))
            except Exception:
                pass

    return cands

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(f"{APP_NAME} — Real Photos + AI Render + Excel Batch")

# Mode
mode = st.sidebar.radio("Mode", ["Real Photos", "AI Render", "Batch from Excel"], index=0)

# Keys (with Secrets fallback)
st.sidebar.subheader("Keys")
gmaps_key_input = st.sidebar.text_input("Google Maps/Places API key", type="password")
serp_key_input  = st.sidebar.text_input("SerpAPI key (optional)", type="password")
openai_key_input = st.sidebar.text_input("OpenAI API key (for AI Render)", type="password")
gmaps_key  = gmaps_key_input  or SECRETS.get("GOOGLE_MAPS_API_KEY", "")
serp_key   = serp_key_input   or SECRETS.get("SERPAPI_KEY", "")
openai_key = openai_key_input or SECRETS.get("OPENAI_API_KEY", "")

# Output
st.sidebar.subheader("Output")
quality = st.sidebar.slider("WebP quality", 60, 95, 82)
make_pin = st.sidebar.checkbox("Also make a Pinterest image (1000×1500)")

# AI settings
st.sidebar.subheader("AI settings")
site = st.sidebar.selectbox("Site style", list(SITE_PROFILES.keys()),
                            index=list(SITE_PROFILES.keys()).index(DEFAULT_SITE))

# LSI controls (AI only)
st.sidebar.caption("LSI expansion is used by AI mode.")
lsi_method = st.sidebar.selectbox("LSI method", ["Heuristic", "Off"], index=0)
images_per_keyword = st.sidebar.number_input("Images per keyword (LSI expansion)", 1, 10, 1)

# Sources (real photos)
st.sidebar.subheader("Sources to use")
use_places_flag = st.sidebar.checkbox("Google Places Photos", value=True, key="use_places_flag")
use_street_flag = st.sidebar.checkbox("Google Street View", value=True, key="use_street_flag")
use_serp_flag = st.sidebar.checkbox("SerpAPI thumbnails (reference only)", value=False, key="use_serp_flag")

# Street View radius
st.sidebar.subheader("Street View")
sv_radius_m = st.sidebar.slider("Search radius (meters)", 25, 500, 250, key="sv_radius_m")

# Invalidate old candidates if source settings change
sources_sig = (use_places_flag, use_street_flag, use_serp_flag, sv_radius_m)
if st.session_state.get("last_sources_sig") != sources_sig:
    st.session_state["last_sources_sig"] = sources_sig
    st.session_state.pop("realphoto_sets", None)

# Session containers
if "realphoto_sets" not in st.session_state:
    st.session_state["realphoto_sets"] = {}
if "zip_items" not in st.session_state:
    st.session_state["zip_items"] = []

col_a, col_b = st.columns(2)

if mode == "Real Photos":
    with col_a:
        st.subheader("Keywords")
        keywords_text = st.text_area(
            "Paste keywords (one per line)",
            "",
            height=180,
            placeholder="Tavern on the Square, Vail Colorado\nBest seafood restaurant in Boston",
        )
        if st.button("Collect candidates"):
            if not gmaps_key:
                st.error("Google Maps/Places API key is required for Places/Street View.")
            else:
                kws = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
                if not kws:
                    st.warning("Please paste at least one keyword.")
                else:
                    st.session_state["realphoto_sets"].clear()
                    prog = st.progress(0.0)
                    total = len(kws)
                    for i, kw in enumerate(kws, start=1):
                        cands = collect_real_photo_candidates(
                            kw,
                            use_places_flag=use_places_flag,
                            use_street_flag=use_street_flag,
                            use_serp_flag=use_serp_flag,
                            sv_radius_m=sv_radius_m,
                            gmaps_key=gmaps_key,
                            serp_key=serp_key
                        )
                        st.session_state["realphoto_sets"][kw] = cands
                        prog.progress(i/total)
        if st.button("Clear"):
            st.session_state["realphoto_sets"].clear()
            st.session_state["zip_items"].clear()
            st.experimental_rerun()

    with col_b:
        st.subheader("Results")
        if not st.session_state["realphoto_sets"]:
            st.info("Run 'Collect candidates' to fetch Google/SerpAPI images.")
        else:
            for kw, cands in st.session_state["realphoto_sets"].items():
                st.markdown(f"#### {kw}")
                if not cands:
                    st.write("No candidates found.")
                    continue
                for idx, c in enumerate(cands, start=1):
                    with st.expander(f"{idx}. {c.title} [{c.source}]"):
                        st.image(c.preview_bytes, use_column_width=True)
                        st.caption(c.license_note)
                        fn = f"{slugify(kw)}_{idx}.webp"
                        if st.button("Create Image", key=f"create_{kw}_{idx}"):
                            try:
                                webp = to_webp_bytes(c.preview_bytes, OUTPUT_W, OUTPUT_H, quality)
                                st.session_state["zip_items"].append((fn, webp))
                                st.success(f"Created {fn}")
                                st.download_button("Download", data=webp, file_name=fn, mime="image/webp",
                                                   key=f"dl_{kw}_{idx}")
                                if make_pin:
                                    pin_fn = f"{slugify(kw)}_{idx}_pinterest.webp"
                                    pin_b = to_webp_bytes(c.preview_bytes, PINTEREST_W, PINTEREST_H, quality)
                                    st.session_state["zip_items"].append((pin_fn, pin_b))
                                    st.download_button("Download Pinterest", data=pin_b, file_name=pin_fn,
                                                       mime="image/webp", key=f"dlp_{kw}_{idx}")
                            except Exception as e:
                                st.error(f"Failed: {e}")

    if st.session_state["zip_items"]:
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fn, b in st.session_state["zip_items"]:
                zf.writestr(fn, b)
        buf.seek(0)
        st.download_button("⬇️ Download all as ZIP", data=buf, file_name="imageforge_realphotos.zip",
                           mime="application/zip")

elif mode == "AI Render":
    size = st.selectbox("OpenAI render size", OPENAI_IMAGE_SIZES, index=0)
    if col_a.button("Generate Image"):
        if not openai_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        else:
            kws = [ln.strip() for ln in keywords_text.splitlines() if ln.strip()]
            if not kws:
                st.warning("Please paste at least one keyword.")
            else:
                outputs: List[Tuple[str, bytes]] = []
                prog = st.progress(0.0)
                total = 0
                for kw in kws:
                    variants = [kw]
                    if images_per_keyword > 1 and lsi_method == "Heuristic":
                        variants = []
                        base_words = kw.split()
                        for i in range(images_per_keyword):
                            extra = f" view {i+1}"
                            variants.append(kw + extra)
                    for v in variants:
                        total += 1

                done = 0
                for kw in kws:
                    variants = [kw]
                    if images_per_keyword > 1 and lsi_method == "Heuristic":
                        variants = []
                        base_words = kw.split()
                        for i in range(images_per_keyword):
                            extra = f" view {i+1}"
                            variants.append(kw + extra)

                    for v in variants:
                        prompt = build_ai_prompt(v, site)
                        try:
                            # Placeholder for your actual OpenAI image call
                            img = Image.new("RGB", (OUTPUT_W, OUTPUT_H), (200, 200, 200))
                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            b = buf.getvalue()
                            webp = to_webp_bytes(b, OUTPUT_W, OUTPUT_H, quality)
                            fn = f"{slugify(v)}.webp"
                            outputs.append((fn, webp))
                        except Exception as e:
                            st.write(f"Error for {v}: {e}")
                        done += 1
                        prog.progress(done/total)

                st.success(f"Generated {len(outputs)} images.")
                import zipfile
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fn, b in outputs:
                        zf.writestr(fn, b)
                buf.seek(0)
                st.download_button("⬇️ Download all as ZIP", data=buf, file_name="imageforge_ai.zip",
                                   mime="application/zip")
    if col_b.button("Clear"):
        st.experimental_rerun()

# -----------------------------
# New: Batch from Excel
# -----------------------------
else:
    st.markdown("### Upload an Excel file with business names")
    st.write(
        "Accepted: `.xlsx` (first row as headers). Common header names: "
        "**name**, **business**, **place**, or choose any column below."
    )

    file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    if file:
        try:
            df = pd.read_excel(file)
        except Exception as e:
            st.error(f"Unable to read Excel: {e}")
            st.stop()

        if df.empty:
            st.warning("The spreadsheet is empty.")
            st.stop()

        # Pick a column for queries
        default_col = None
        for c in df.columns:
            if str(c).strip().lower() in ("name", "business", "place", "query", "keyword"):
                default_col = c
                break
        column = st.selectbox(
            "Select the column that contains business names/queries:",
            list(df.columns),
            index=(list(df.columns).index(default_col) if default_col in df.columns else 0),
        )

        # Optional: city/state context to improve disambiguation
        context_hint = st.text_input(
            "Optional location/context to append to each query (e.g., 'Vail, Colorado')",
            "",
        )

        # Batch options
        col1, col2 = st.columns(2)
        images_per_business = col1.number_input(
            "Images per business (saved to ZIP)", 1, 12, 3
        )
        use_street_in_batch = col2.checkbox(
            "Also try Street View if no Places Photo", value=True
        )

        start = st.button("Run batch (Create images + ZIP)")
        if start:
            if not gmaps_key:
                st.error("Google Maps/Places API key is required (enter in sidebar).")
                st.stop()

            import zipfile
            zip_buf = io.BytesIO()
            created = 0
            prog = st.progress(0.0)

            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                rows = df[column].astype(str).tolist()
                total = len(rows)
                for i, raw in enumerate(rows, start=1):
                    q = raw.strip()
                    if not q:
                        prog.progress(i/total)
                        continue
                    if context_hint:
                        q = f"{q}, {context_hint}"

                    # Collect candidates (this may return multiple Places photos + Street View)
                    cands = collect_real_photo_candidates(
                        q,
                        use_places_flag=True,
                        use_street_flag=use_street_in_batch,
                        use_serp_flag=False,             # keep Serp off in batch
                        sv_radius_m=sv_radius_m,
                        gmaps_key=gmaps_key,
                        serp_key=None,
                    )

                    # Separate Places and Street View
                    places = [c for c in cands if "Places Photo" in c.title]
                    street = [c for c in cands if "Street View" in c.title]

                    # Build a pool: prefer Places; fall back to Street View
                    pool = places
                    if not pool and use_street_in_batch:
                        pool = street

                    # Limit how many we actually save per business
                    pool = pool[: int(images_per_business)]

                    if not pool:
                        prog.progress(i/total)
                        continue

                    base = slugify(raw)

                    for j, chosen in enumerate(pool, start=1):
                        try:
                            webp = to_webp_bytes(chosen.preview_bytes, OUTPUT_W, OUTPUT_H, quality)
                            fn = f"{base or 'image'}_{i}_{j}.webp"
                            zf.writestr(fn, webp)
                            created += 1
                            if make_pin:
                                pin_b = to_webp_bytes(chosen.preview_bytes, PINTEREST_W, PINTEREST_H, quality)
                                zf.writestr(f"{base or 'image'}_{i}_{j}_pinterest.webp", pin_b)
                        except Exception as e:
                            st.write(f"⚠️ {raw} (image #{j}): {e}")

                    prog.progress(i/total)

            zip_buf.seek(0)
            st.success(f"Done. Created {created} images.")
            st.download_button("⬇️ Download batch as ZIP", data=zip_buf,
                               file_name="imageforge_batch.zip", mime="application/zip")
