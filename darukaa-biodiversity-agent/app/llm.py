"""Optional LLM layer. The system is fully functional without it.

The LLM is only allowed to (1) extract fields from messy text and (2) write a short summary of
recommendations that the rule engine ALREADY produced. A grounding guard rejects any summary
containing a number that is not in the cited evidence cards or the user's own data, which blocks
the classic failure of inventing effect sizes and 'FAO says X%' style citations."""
from __future__ import annotations

import json
import os
import re

MODEL = os.getenv("LLM_MODEL", "claude-sonnet-5")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def enabled() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY")) and os.getenv("DISABLE_LLM") != "1"


def _call(system: str, user: str, max_tokens: int = 600) -> str:
    import anthropic
    msg = anthropic.Anthropic().messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                                                messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


EXTRACT_SYS = ("Extract site variables from the user's text. Return ONLY a JSON object using these keys when "
               "explicitly stated: soc_pct, ph, soil_moisture_class(low|medium|high), rainfall_mm, "
               "rainfall_class(low|medium|high), mean_temp_c, climate(arid|semi-arid|sub-humid|humid|temperate|tropical), "
               "land_use(cropland|pasture|forest|degraded|urban|wetland), crop, crop_system(monoculture|rotation|intercrop), "
               "native_habitat_pct, species_richness, biodiversity_trend(declining|stable|increasing), "
               "fragmentation(low|medium|high), pesticide_use(low|medium|high), pollution(low|medium|high), "
               "deforestation(bool), near_water_body(bool), lat, lon. Never guess values that are not stated.")


def extract(text: str) -> dict:
    if not enabled():
        return {}
    try:
        raw = _call(EXTRACT_SYS, text, 300)
        return json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
    except Exception:
        return {}


def allowed_numbers(recs: list[dict], evidence: dict, profile: dict) -> set[str]:
    allowed = set()
    for r in recs:
        allowed |= set(NUM_RE.findall(r["expected_effect"]))
        for c in r["citations"]:
            card = evidence[c["id"]]
            allowed |= set(NUM_RE.findall(card["finding"])) | {str(card["year"])}
    for v in profile.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            allowed |= {str(v), f"{v:g}"}
    return allowed


def grounded(text: str, allowed: set[str]) -> tuple[bool, list[str]]:
    bad = [n for n in NUM_RE.findall(text) if n not in allowed and n not in {"1", "2", "3", "4", "5"}]
    return (not bad), bad


def summarize(recs: list[dict], evidence: dict, profile: dict) -> tuple[str | None, str | None]:
    if not enabled() or not recs:
        return None, None
    sys = ("You are an environmental scientist. Summarize the given recommendations for a land manager in "
           "<=90 words, plain language. Use ONLY numbers that appear in the input. Do not add sources, "
           "percentages or claims that are not in the input.")
    try:
        text = _call(sys, json.dumps({"profile": profile, "recommendations": recs})[:12000], 250).strip()
    except Exception as e:
        return None, f"LLM summary skipped ({type(e).__name__})"
    ok, bad = grounded(text, allowed_numbers(recs, evidence, profile))
    if not ok:
        return None, f"LLM summary rejected by grounding guard (unsupported numbers: {', '.join(bad)})"
    return text, None
