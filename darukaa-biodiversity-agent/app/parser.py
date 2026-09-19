"""Turns free text or loosely-structured JSON into a SiteProfile. Deterministic (regex + synonyms),
so it works with no LLM and is fully testable. An optional LLM extractor can be layered on top."""
from __future__ import annotations

import re
from typing import Any

from .models import SiteProfile

NUM = r"(\d+(?:\.\d+)?)"
LEVEL_WORDS = {
    "very low": "low", "low": "low", "poor": "low", "scarce": "low", "scanty": "low", "dry": "low",
    "moderate": "medium", "medium": "medium", "average": "medium",
    "high": "high", "heavy": "high", "good": "high", "abundant": "high", "intensive": "high",
}
CLIMATES = ["semi-arid", "semi arid", "semiarid", "arid", "sub-humid", "humid", "temperate", "tropical"]
CROPS = ["wheat", "rice", "paddy", "maize", "corn", "sugarcane", "cotton", "soybean", "millet", "sorghum",
         "barley", "potato", "tea", "coffee"]


def _level(text: str) -> str | None:
    for w in sorted(LEVEL_WORDS, key=len, reverse=True):
        if re.search(rf"\b{w}\b", text):
            return LEVEL_WORDS[w]
    return None


def _near(t: str, keyword: str, window: int = 30) -> str | None:
    m = re.search(rf"{keyword}.{{0,{window}}}", t)
    return m.group(0) if m else None


def parse_text(text: str) -> SiteProfile:
    t = text.lower().replace("–", "-")
    d: dict[str, Any] = {}

    m = re.search(rf"(?:soil organic carbon|organic carbon|\bsoc\b)[^0-9%]{{0,25}}{NUM}\s*%", t) \
        or re.search(rf"{NUM}\s*%\s*(?:soil organic carbon|organic carbon|soc\b)", t)
    if m:
        d["soc_pct"] = float(m.group(1))

    m = re.search(rf"\bph\b[^0-9]{{0,12}}{NUM}", t)
    if m and 2 <= float(m.group(1)) <= 11:
        d["ph"] = float(m.group(1))

    m = re.search(rf"{NUM}\s*mm", t)
    if m and re.search(r"rain|precip|mm/y|mm per year|annual", t):
        d["rainfall_mm"] = float(m.group(1))
    else:
        seg = _near(t, r"rain(?:fall)?", 25) or _near(t, r"precipitation", 25)
        pre = re.search(r"(low|scanty|poor|erratic|moderate|high|heavy|good)\s+rain", t)
        lv = (pre and LEVEL_WORDS.get(pre.group(1), "low")) or (seg and _level(seg))
        if lv:
            d["rainfall_class"] = lv
        elif re.search(r"drought|dry ?land", t):
            d["rainfall_class"] = "low"

    m = re.search(rf"(-?{NUM})\s*(?:°|deg(?:rees)?)?\s*c\b", t)
    if m and re.search(r"temp|°|deg|hot|warm|cold", t):
        d["mean_temp_c"] = float(m.group(1))

    seg = _near(t, r"(?:soil )?moisture", 20)
    if seg:
        mm = re.search(rf"moisture[^0-9]{{0,10}}{NUM}\s*%", t)
        if mm:
            v = float(mm.group(1))
            d["soil_moisture_class"] = "low" if v < 15 else "medium" if v < 30 else "high"
        elif _level(seg):
            d["soil_moisture_class"] = _level(seg)

    for c in CLIMATES:
        if c in t:
            d["climate"] = "semi-arid" if "semi" in c else c
            break

    if re.search(r"monocultur|mono-crop|monocrop|single crop", t):
        d["crop_system"] = "monoculture"
    elif re.search(r"intercrop", t):
        d["crop_system"] = "intercrop"
    elif re.search(r"rotation", t):
        d["crop_system"] = "rotation"

    for c in CROPS:
        if re.search(rf"\b{c}\b", t):
            d["crop"] = c
            break

    if re.search(r"\bforest\b(?! loss)|woodland", t) and not re.search(r"deforest|agroforest", t):
        d["land_use"] = "forest"
    if re.search(r"crop|farm|field|agricultur|wheat|rice|maize|cultivat", t):
        d["land_use"] = "cropland"
    elif re.search(r"pasture|grazing|rangeland|grassland", t):
        d["land_use"] = "pasture"
    elif re.search(r"degraded|barren|wasteland|eroded", t):
        d["land_use"] = "degraded"

    m = re.search(rf"{NUM}\s*%\s*(?:of the land |of land )?(?:is )?(?:native|natural|semi-natural)\s*(?:habitat|vegetation|cover)", t) \
        or re.search(rf"(?:native|natural) (?:habitat|vegetation|cover)[^0-9]{{0,15}}{NUM}\s*%", t)
    if m:
        d["native_habitat_pct"] = float(m.group(1))

    m = re.search(r"(\d+)\s*(?:plant |bird |insect )?species", t)
    if m:
        d["species_richness"] = int(m.group(1))
    if re.search(r"declin|decreas|los(?:s|ing)|disappear|fewer", t) and re.search(r"biodiversity|species|bird|insect|pollinator|bee", t):
        d["biodiversity_trend"] = "declining"

    if re.search(r"fragment|isolated patch|patches", t):
        d["fragmentation"] = _level(_near(t, "fragment", 25) or "") or "high"
    if re.search(r"deforest|clear(?:ed|ing) (?:of )?(?:trees|forest)|tree cutting|logging", t):
        d["deforestation"] = True
    seg = _near(t, r"(?:pesticide|insecticide|herbicide|fungicide)s?", 25)
    if seg:
        pre = re.search(r"(heavy|high|intensive|moderate|low|little)\s+(?:use of )?(?:pesticide|insecticide|chemical)", t)
        d["pesticide_use"] = LEVEL_WORDS.get(pre.group(1), "medium") if pre else (_level(seg) or "medium")
    seg = _near(t, r"pollut", 25)
    if seg:
        d["pollution"] = _level(seg) or "medium"
    if re.search(r"river|stream|canal|lake|pond|wetland|drain", t):
        d["near_water_body"] = True

    m = re.search(r"(-?\d{1,2}\.\d+)\s*[, ]\s*(-?\d{1,3}\.\d+)", t)
    if m:
        d["lat"], d["lon"] = float(m.group(1)), float(m.group(2))

    return SiteProfile(**d, provenance={k: "user" for k in d})


ALIASES = {
    "soil_organic_carbon": "soc_pct", "soc": "soc_pct", "organic_carbon": "soc_pct",
    "soil_ph": "ph", "rainfall": "rainfall", "precipitation": "rainfall", "rain": "rainfall",
    "temperature": "mean_temp_c", "temp_c": "mean_temp_c", "moisture": "soil_moisture_class",
    "soil_moisture": "soil_moisture_class", "region": "climate", "climate_zone": "climate",
    "crop": "crop", "latitude": "lat", "longitude": "lon", "native_habitat": "native_habitat_pct",
    "pesticides": "pesticide_use", "landuse": "land_use", "land_cover": "land_use",
}


def parse_structured(obj: dict[str, Any]) -> SiteProfile:
    """Accepts clean JSON (field names of SiteProfile) or looser keys/values, e.g.
    {"soil_organic_carbon": "0.3%", "rainfall": "low", "crop": "monoculture wheat", "region": "semi-arid"}."""
    d: dict[str, Any] = {}
    loose_text = []
    for k, v in obj.items():
        key = ALIASES.get(k.lower().strip().replace(" ", "_"), k.lower().strip().replace(" ", "_"))
        if v is None:
            continue
        if isinstance(v, str):
            loose_text.append(f"{k.replace('_', ' ')}: {v}")
            vs = v.strip().lower()
            num = re.match(rf"^-?{NUM}", vs.replace("%", "").strip())
            if key == "rainfall":
                if num and "mm" in vs or (num and float(num.group(0)) > 20):
                    d["rainfall_mm"] = float(num.group(0))
                else:
                    d["rainfall_class"] = LEVEL_WORDS.get(vs, _level(vs) or "medium")
                continue
            if key in {"soc_pct", "ph", "mean_temp_c", "native_habitat_pct", "lat", "lon", "rainfall_mm"} and num:
                d[key] = float(num.group(0))
                continue
            if key in {"soil_moisture_class", "pesticide_use", "pollution", "fragmentation", "rainfall_class"}:
                d[key] = LEVEL_WORDS.get(vs, _level(vs) or vs)
                continue
            if key == "crop":
                p = parse_text(vs)
                d["crop"] = p.crop or v
                if p.crop_system:
                    d["crop_system"] = p.crop_system
                continue
            if key == "climate":
                p = parse_text(vs)
                if p.climate:
                    d["climate"] = p.climate
                else:
                    d["region"] = v
                continue
        elif key == "rainfall":
            d["rainfall_mm"] = float(v)
            continue
        if key in SiteProfile.model_fields and key != "provenance":
            d[key] = v
    # let the text parser pick up anything structured keys implied (e.g. "monoculture wheat")
    implied = parse_text(". ".join(loose_text)).model_dump(exclude_none=True, exclude={"provenance"})
    for k, v in implied.items():
        d.setdefault(k, v)
    return SiteProfile(**d, provenance={k: "user" for k in d})
