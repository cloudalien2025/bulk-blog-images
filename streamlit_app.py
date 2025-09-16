# streamlit_app.py
import base64
import io
import random
import re
import zipfile
from typing import List

import requests
from PIL import Image
import streamlit as st

# -------------------- App setup --------------------
st.set_page_config(page_title="Bulk Blog Image Generator", layout="wide")

# Optional simple team password (set in Secrets -> APP_PASSWORD or hardcode here)
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")  # "" disables the gate

SITE_PROFILES = {
    "vailvacay.com":  "Photorealistic alpine resort & village scenes in the Colorado Rockies; ski terrain; evergreen forests; cozy lodges; no text.",
    "bangkokvacay.com":"Photorealistic Southeast Asian city scenes; temples, canals, street food markets, tuk-tuks; golden-hour light; no text.",
    "bostonvacay.com": "Photorealistic New England city imagery; Beacon Hill brownstones, harbor, Charles River, fall foliage; no text.",
    "ipetzo.com":      "Photorealistic pet lifestyle images; dogs/cats with owners outdoors/indoors; neutral interiors/parks; no brands; no text.",
    "1-800deals.com":  "Photorealistic retail/ecommerce visuals; shopping scenes, parcels, generic products; clean backgrounds; no brands; no text.",
}
DEFAULT_SITE = "vailvacay.com"

API_IMAGE_SIZE = "1536x1024"      # Supported: 1024x1024, 1024x1536, 1536x1024, or "auto"
OUTPUT_W, OUTPUT_H = 1200, 675    # Final blog size
DEFAULT_QUALITY = 82              # WebP quality (70–82 is good)

# -------------------- Utility helpers --------------------
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

# ================= UNIVERSAL SMART PROMPT PLANNER =================
def detect_season(keyword: str) -> str:
    k = keyword.lower()
    if any(w in k for w in ["winter", "snow", "dec", "jan", "feb", "christmas"]): return "winter"
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
            "eye-level camera, 24–35mm FoV, subtle leading lines from a path/road/water.")

def negatives(site: str) -> str:
    common = "No typography overlay; avoid readable signs/brand logos; avoid illegal/explicit content."
    if site == "1-800deals.com":
        return common + " Packaging must be generic with blank labels; avoid trademarks/SKUs/barcodes."
    if site == "ipetzo.com":
        return common + " No harmful gear or unsafe handling."
    if site == "vailvacay.com":
        return common + " Avoid generic condo rows unless the keyword is lodging."
    if site == "bostonvacay.com":
        return common + " Prefer generic wayfinding over specific transit/airline names."
    if site == "bangkokvacay.com":
        return (common + " Nightlife: exterior/streetscape or cabaret entrance only; no lewd scenes, "
                "no sexual acts, no suggestive close-ups, no drug/vape brand marks.")
    return common

def classify(keyword: str) -> set:
    k = keyword.lower()
    tags = set()

    # Universal intents
    if "things to do" in k or "what to do" in k: tags.add("todo")
    if any(w in k for w in ["how far", "distance", "drive", "driving time", "road trip", "get to", "directions"]): tags.add("drive")
    if any(w in k for w in ["where to stay", "hotel", "lodging", "resort"]): tags.add("hotel")
    if any(w in k for w in ["hike", "trail"]): tags.add("hike")
    if any(w in k for w in ["family", "kids", "kid", "stroller", "with children", "non skiers", "non-skiers"]): tags.add("family")
    if any(w in k for w in ["compare", "compared to", " vs "]): tags.add("compare")
    if any(w in k for w in ["why is", "expensive", "pricey", "how much", "cost of", "costs"]): tags.add("costly")
    if any(w in k for w in ["winter","december","january","february","summer","july","august","rain","monsoon","september","november","may","october","thanksgiving","memorial day","christmas"]): tags.add("seasonal")

    # Vail specifics
    if any(w in k for w in ["back bowls", "mongolia bowl"]): tags.add("back_bowls")
    if "gondola" in k or "ski lift" in k: tags.add("gondola")
    if any(w in k for w in ["ticket", "tickets", "price", "discount", "free", "after 3", "after 3:30"]): tags.add("tickets")
    if "lionshead" in k: tags.add("lionshead")
    if "cigar" in k: tags.add("cigar")
    if any(w in k for w in ["altitude", "highest elevation"]): tags.add("altitude")
    if any(w in k for w in ["moose", "wildlife"]): tags.add("moose")
    if any(w in k for w in ["headquarters", "corporate"]): tags.add("corporate")
    if any(w in k for w in ["lindsey vonn", "lindsay vonn"]): tags.add("celebrity")
    if any(w in k for w in ["ski patrol"]): tags.add("skipatrol")
    if any(w in k for w in ["budget", "on a budget", "cheap"]): tags.add("budget")
    if any(w in k for w in ["burger", "best burger", "restaurants", "restaurant", "cafe", "breakfast", "westside cafe"]): tags.add("food")
    # Spanish support (Vail)
    if any(w in k for w in ["qué hacer", "que hacer"]): tags.add("todo")
    if any(w in k for w in ["niños", "ninos"]): tags.add("family")

    # Boston transit / districts / anchors
    if "ferry" in k or "cat ferry" in k or "ferries" in k: tags.add("ferry")
    if "train" in k or "amtrak" in k: tags.add("train")
    if any(w in k for w in ["flight", "flights", "flight time", "airport", "logan", "bos"]): tags.add("flight")
    if "cruise" in k or "boat tour" in k or "harbor cruise" in k: tags.add("cruise")
    if "aquarium" in k: tags.add("aquarium")
    if "north station" in k: tags.add("north_station")
    if "south station" in k: tags.add("south_station")
    if "faneuil" in k or "quincy market" in k: tags.add("faneuil")
    if "back bay" in k or "park plaza" in k: tags.add("back_bay")
    if "seaport" in k or "cruise terminal" in k or "black falcon" in k: tags.add("seaport")
    if "bar harbor" in k or "acadia" in k: tags.add("bar_harbor")
    if any(w in k for w in ["nearest airport to bar harbor", "bangor", "ellsworth", "hancock county"]): tags.add("bar_harbor_air")
    if "maine" in k: tags.add("maine_trip")
    if any(w in k for w in ["morning", "early morning"]): tags.add("morning")
    if any(w in k for w in ["birthday", "bday", "freebies", "free stuff", "perks", "free admission"]): tags.add("birthday")
    if any(w in k for w in ["persian", "iranian"]): tags.add("persian")
    if any(w in k for w in ["nerdy", "geeky"]): tags.add("nerdy")
    if any(w in k for w in ["burger", "best burger", "restaurants", "restaurant", "cafe", "breakfast"]): tags.add("food")

    # Bangkok transit / districts / nightlife / shopping
    if "bts" in k: tags.add("bts")
    if "mrt" in k: tags.add("mrt")
    if any(w in k for w in ["airport", "bkk", "suvarnabhumi", "don mueang", "dmk", "customs"]): tags.add("airport")
    if any(w in k for w in ["yaowarat", "chinatown"]): tags.add("yaowarat")
    if "silom" in k: tags.add("silom")
    if "asoke" in k or "asok" in k: tags.add("asoke")
    if "sukhumvit" in k: tags.add("sukhumvit")
    if "ari" in k: tags.add("ari")
    if "khao san" in k: tags.add("khaosan")
    if any(w in k for w in ["red light", "naughty", "freelancer", "hooker", "short time"]): tags.add("nightlife")
    if "ping pong show" in k or "pingpong" in k: tags.add("ping_pong")
    if "soapy" in k or "nuru" in k or "body to body" in k: tags.add("soapy")
    if "ladyboy" in k or "kathoey" in k or "lady boy" in k: tags.add("ladyboy")
    if any(w in k for w in ["nana plaza", "soi cowboy", "patpong"]): tags.add("bar_zone")
    if "vape" in k or "relx" in k: tags.add("vape")
    if "passport photo" in k or "id photo" in k or "visa photo" in k: tags.add("passport_photo")
    if "wacoal" in k or "nike" in k or "fake designer" in k or "cheap electronics" in k or "electronics cheap" in k: tags.add("shopping")
    if any(w in k for w in ["seafood", "sushi", "best seafood", "restaurant", "where to eat"]): tags.add("food")
    if "jessica bangkok" in k: tags.add("adult_person_query")

    return tags

def choose_one(options): 
    return random.choice(options)

def geo_enrichment(site: str, keyword: str, season: str, tags: set) -> str:
    s = site.lower()
    k = keyword.lower()

    # ----- Vail Vacay (Colorado corridor) -----
    if s == "vailvacay.com":
        # corridor: Denver <-> Vail
        if "between vail and denver" in k:
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

        # Regional drives (Aspen/Breck etc.)
        if "drive" in tags:
            return choose_one([
                "I-70 mountain approach framed by evergreens and snowcapped peaks; safe roadside vantage",
                "Glenwood Canyon elevated highway above the Colorado River with sheer cliffs",
                "Independence Pass alpine switchbacks with jagged peaks and roadside wildflowers"
            ])

        # Back Bowls
        if "back_bowls" in tags:
            return "broad snowy panorama of Vail's open bowls from a high ridge, untouched powder fields, dramatic sky"

        # Gondola / ticketing / free-after-3:30
        if "gondola" in tags:
            if "tickets" in tags:
                return "gondola base area with ticket windows and queue ropes, cabins arriving; late-afternoon light; no readable signs"
            return "Eagle Bahn gondola cabins gliding above the village with peaks behind; dynamic angle"
        if "tickets" in tags:
            return "lift plaza scene implying ticketing: skiers near gates and scanners; no readable screens or text"

        # Food (Lionshead / burgers / cafes)
        if "food" in tags:
            if "lionshead" in tags:
                return "al-fresco table in Lionshead square with the gondola in background; plated burger and fries (no logos)"
            return choose_one([
                "gourmet burger on a wooden table by a window with mountain view, shallow depth of field (no branding)",
                "cozy alpine gastropub interior with burgers and local-style sides under warm lighting (no logos)"
            ])

        # Romantic
        if "romantic" in k:
            return choose_one([
                "evening sleigh ride scene with blankets under string lights and snowy pines",
                "riverside path at dusk with warm lanterns and distant lodge glow; couple in silhouette"
            ])

        # Family/kids / non-skiers / indoor activities
        if "family" in tags:
            return choose_one([
                "easy riverside trail with strollers and kids; aspens and peaks around",
                "lakeside path with playground elements and families; gentle mountains behind",
                "indoor activity center with climbing/bouldering walls; bright light; no logos"
            ])

        # Moose / wildlife
        if "moose" in tags:
            return "broad meadow near treeline at dawn with a distant moose by a creek and alpine peaks beyond"

        # Expensive / costly
        if "costly" in tags:
            return choose_one([
                "slopeside luxury hotel valet with a row of high-end SUVs and bell staff, skis and boot bags around",
                "après-ski terrace with champagne buckets on ice, fire pits and heat lamps, mountain view",
                "private jets lined up on the tarmac at Eagle County Regional Airport with snowy peaks behind",
                "real-estate sales lounge with scale model and panoramic windows, mountain backdrop (no logos)"
            ])

        # Altitude / highest elevation
        if "altitude" in tags:
            return "high alpine ridge above treeline with wind-swept snow cornices and distant fourteeners; sense of thin air"

        # Weather / seasonal
        if "seasonal" in tags and "snow" in k:
            return "weather-forward landscape over Vail: sun breaks through snow showers over peaks and valley, dramatic clouds"

        # Corporate HQ
        if "corporate" in tags:
            return "modern mountain-town office building with glass and timber; lobby glow; no company wordmarks"

        # Celebrity (privacy-safe)
        if "celebrity" in tags:
            return "discreet high-end neighborhood on a snowy slope: gated driveway, evergreens, modern mountain home; no people, no signage"

        # Ski patrol (generic)
        if "skipatrol" in tags:
            return "patroller viewpoint: red toboggan and rescue gear near a ridge, sweeping slopes beyond; no logos"

        # Cigar lounge
        if "cigar" in tags:
            return "upscale cigar lounge interior with leather chairs, stone fireplace, amber lighting, humidor wall; no branding"

        # Budget vibes
        if "budget" in tags:
            return "free or low-cost activities: village bus stop with skiers and gear, or a picnic by Gore Creek; friendly vibe"

        # Fallback for generic “Vail Colorado” etc.
        return "place-relevant alpine hero scene with Gore Creek, evergreen forest, and ski slopes beyond; balanced, editorial feel"

    # ----- Boston Vacay -----
    if s == "bostonvacay.com":
        # Bar Harbor / Acadia focus
        if "bar_harbor" in tags and "ferry" in tags:
            return "high-speed catamaran ferry at a Maine harbor terminal with boarding ramp and granite shoreline; no readable vessel names"
        if "bar_harbor" in tags and "train" in tags:
            return "North Station concourse with travelers headed for a Downeaster-like service; generic signs only"
        if "bar_harbor" in tags and ("flight" in tags or "bar_harbor_air" in tags):
            return "small regional Maine airport scene with a turboprop on the apron, pine trees beyond; no airline logos"
        if "bar_harbor" in tags and "drive" in tags:
            return choose_one([
                "coastal Route 1 drive past a lighthouse on a rocky headland at golden hour",
                "Acadia National Park scenic loop road with granite cliffs and Atlantic surf"
            ])

        # General ferries/trains/flights/cruise from Boston
        if "ferry" in tags:
            return "Boston Harbor ferry boarding at a downtown pier with skyline and harbor islands; gangway and crew; no readable signs"
        if "train" in tags and "north_station" in tags:
            return "North Station concourse with portal to tracks; commuters in motion blur; generic wayfinding"
        if "train" in tags and "south_station" in tags:
            return "South Station great hall with arched windows and departures board implied; no readable text"
        if "flight" in tags:
            return choose_one([
                "Logan Airport terminal interior at dusk with planes at the gates outside; generic counters, no logos",
                "airplane wing over the Atlantic at sunrise departing Boston"
            ])
        if "cruise" in tags or "seaport" in tags:
            return "large cruise ship docked at Flynn Cruiseport (Seaport) with modern skyline behind; golden-hour reflections"

        # Neighborhood anchors
        if "faneuil" in tags:
            return "Faneuil/Quincy Market plaza with cobblestones and food halls; lively but no readable signage"
        if "back_bay" in tags:
            return choose_one([
                "Newbury Street cafe tables under mature trees; brownstones lining the block",
                "Commonwealth Avenue Mall with canopy of trees and classic brownstones"
            ])
        if "north_station" in tags and "train" not in tags:
            return "North Station exterior concourse area with arena facade implied; urban bustle; generic wayfinding"
        if "south_station" in tags and "train" not in tags:
            return "South Station facade with columns and taxis; commuters crossing the plaza"

        # Morning / early morning
        if "morning" in tags:
            return choose_one([
                "sunrise on the Charles River Esplanade with rowers and runners, soft pink sky",
                "early bakery coffee line with pastries in warm window light; no branding"
            ])

        # Birthday freebies (editorial vibe, no logos)
        if "birthday" in tags:
            return choose_one([
                "cafe server placing a complimentary dessert with a candle on a small table; cozy interior; no menus readable",
                "host handing a birthday perk at a museum/aquarium admissions area; friendly gesture; no readable text"
            ])

        # Food (including Persian; burger)
        if "food" in tags and "persian" in tags:
            return "cozy Persian restaurant table with kebab platter, saffron rice, grilled tomatoes, patterned tile details; no logos"
        if "food" in tags and ("burger" in k or "best burger" in k):
            return "gourmet burger with fries near a window overlooking brick streets; shallow depth of field; no branding"
        if "food" in tags:
            return "classic Boston eatery interior with warm wood and pendant lights; plated comfort food; no logos"

        # Nerdy / geeky
        if "nerdy" in tags:
            return choose_one([
                "interactive science museum gallery with kinetic exhibits and visitors, bright lighting",
                "board-game cafe interior with shelves of games, small groups playing; no recognizable boxes"
            ])

        # Family / kids / aquarium nearby
        if "family" in tags and "aquarium" in tags:
            return "New England Aquarium exterior with harbor and glass facade; families arriving; no readable signage"
        if "aquarium" in tags:
            return "harborfront view with a modern glass-fronted aquarium building and seals pool; people small in frame"

        # Default Boston hero scenes
        return choose_one([
            "Beacon Hill brownstones with gas lamps and brick sidewalks",
            "Boston Public Garden footbridge with swan boats",
            "Charles River Esplanade with sailboats and skyline at golden hour",
            "Harborwalk with skyline across the harbor"
        ])

    # ----- Bangkok Vacay (safe, brand-friendly) -----
    if s == "bangkokvacay.com":
        # Getting to Chinatown (Yaowarat) by BTS/MRT
        if "yaowarat" in tags and ("bts" in tags or "mrt" in tags):
            return choose_one([
                "elevated train platform with a train arriving and generic signs; evening light; city in background",
                "street-level exit from a skytrain station onto a lantern-lined Yaowarat road; signs abstracted"
            ])
        if "yaowarat" in tags:
            return "lantern-lit food street with sizzling woks and neon reflections after rain; shallow depth of field"

        # Airports / customs / airport-to-city
        if "airport" in tags and "drive" in tags:
            return "Airport Rail Link platform with a sleek train arriving; modern concourse; generic wayfinding"
        if "airport" in tags:
            return choose_one([
                "Suvarnabhumi-like terminal interior with sweeping steel arches and glass; planes at gate; no airline logos",
                "Don Mueang-like terminal concourse with travelers and tropical light; no readable branding"
            ])

        # BTS / MRT generally
        if "mrt" in tags:
            return "ticket vending area with touchscreens and turnstiles; people tapping cards; branding abstract"
        if "bts" in tags:
            return "elevated BTS tracks curving over a boulevard with skyline and palms; golden-hour light"

        # Adult / nightlife handled safely
        if "adult_person_query" in tags:
            return "nightlife district streetscape with neon and crowds in motion blur; exterior only; generic marquees"
        if "ping_pong" in tags or "bar_zone" in tags or "nightlife" in tags:
            return choose_one([
                "neon-lit alley with bars and crowds; exterior-only view; signs defocused",
                "cabaret theatre entrance with plumes/feathers poster style, velvet ropes, lights; no faces; no text"
            ])
        if "ladyboy" in tags:
            return choose_one([
                "cabaret foyer with stage lights and sequined costumes on mannequins; inclusive mood; no people",
                "street mural with rainbow motif near entertainment district; night ambience"
            ])
        if "soapy" in tags:
            return "spa exterior with frosted-glass door and warm light spilling onto sidewalk; no people; signage abstract"

        # Vape / shopping / passport photos
        if "vape" in tags:
            return "generic electronics kiosk shelves with unbranded vape-style devices; neutral lighting; no logos"
        if "passport_photo" in tags:
            return "small photo studio interior: neutral backdrop, softbox lights, camera on tripod; tidy and unbranded"
        if "shopping" in tags and "wacoal" in k:
            return "department store lingerie section with neutral mannequins and soft lighting; no brand marks"
        if "shopping" in tags and "nike" in k:
            return "athletic shoe wall display with generic sneakers in a bright store; no logos"
        if "shopping" in tags:
            return "indoor market lane with clothing and electronics stalls; lively but unbranded; no counterfeit cues"

        # Food
        if "food" in tags and "sukhumvit" in tags and "seafood" in k:
            return "Sukhumvit restaurant interior with iced seafood platter and herbs on marble; modern decor; no logos"
        if "food" in tags and "sushi" in k:
            return "minimal sushi counter with chef hands plating nigiri; wooden counter; no brand markers"
        if "food" in tags:
            return "night market grill stall with skewers and steam under tungsten lights; shallow depth of field"

        # District anchors
        if "silom" in tags:
            return "Silom streetscape with skytrain overhead and bustling sidewalks; leafy median; dusk glow"
        if "asoke" in tags or "sukhumvit" in tags:
            return "busy Sukhumvit intersection with BTS tracks above and city lights; traffic trails; pedestrians"
        if "ari" in tags:
            return "Ari neighborhood cafe street with trees and small shops; relaxed, creative vibe"

        # Seasonal cues
        if "seasonal" in tags and "september" in k:
            return "after-rain reflections on a Chinatown alley with lanterns and steam; umbrellas, neon, puddles"
        if "seasonal" in tags and any(m in k for m in ["december","january"]):
            return "cool season sunset from a riverside promenade with clear air and soft pastel sky"

        # LGBTQ+ / gay life (safe, inclusive)
        if "gay" in k:
            return "rainbow-lit rooftop bar with friends chatting over the skyline; inclusive atmosphere; no text"

        # Equator curiosity
        if "equator" in k:
            return "public plaza with a large metallic globe sculpture and city skyline behind; editorial travel feel"

        # Default Bangkok hero scenes
        return choose_one([
            "temple courtyard with ornate roofs and palms in warm light",
            "canal scene with wooden houses and long-tail boat passing",
            "neon night market lane with steam and crowds; signs abstracted"
        ])

    # ----- iPetzo (pets) -----
    if s == "ipetzo.com":
        if "grooming" in k:
            return "bright grooming salon scene with a dog being trimmed on a table; tidy tools; no branding"
        if "training" in k:
            return "obedience training in a park; handler rewarding a dog with treats; correct leash handling"
        if "vet" in k:
            return "modern vet clinic exam room; vet gently checking a pet; clean, unbranded space"
        if "pet travel" in k:
            return "pet-friendly hotel room or car with a safely harnessed dog; travel gear without logos"
        if "cat" in k:
            return "sunny living-room window light with a cat on a couch or cat tree; tasteful minimal decor"
        return "owners walking a dog on a leafy path or city park; friendly mood; neutral collars with no logos"

    # ----- 1-800deals (retail) -----
    if s == "1-800deals.com":
        if any(t in tags for t in ["retail","todo"]):
            return choose_one([
                "unboxing on a clean table with plain cardboard box and tissue (no logos)",
                "shopping cart close-up in a bright generic aisle with abstract products (no logos)",
                "parcel stack at a doorstep with neutral labels and tape (no branding)"
            ])
        if "costly" in tags:
            return "premium unboxing with elegant plain packaging, tissue and ribbon (no logos)"
        return "generic ecommerce product lifestyle scene with plain packaging and soft studio light"

    # Fallback for unknown sites
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
# ================= END PROMPT PLANNER =================

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
st.caption("Paste keywords (one per line). The app plans a scene per keyword, generates a DALL·E image, crops to 1200×675, and bundles a ZIP.")

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
    height=260,
    placeholder=(
        "Why is Vail so expensive\n"
        "Things to do between Denver and Vail\n"
        "BTS to Chinatown Bangkok\n"
        "Boston to Bar Harbor ferry"
    ),
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
