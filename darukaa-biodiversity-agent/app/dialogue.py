"""Conversation manager: merges each turn into the session's site profile (memory), asks targeted
clarifying questions until at least 3 variable families are known, then calls the reasoning engine."""
from __future__ import annotations

import re

from pydantic import ValidationError

from . import geo, llm, store
from .knowledge import get_retriever, load_kb
from .models import ChatRequest, ChatResponse, SiteProfile
from .parser import parse_structured, parse_text
from .reasoning import recommend

MIN_GROUPS = 3

# (group, field(s) that satisfy it, question, why it matters)
QUESTIONS = [
    ("soil", ["soc_pct"], "What is the soil organic carbon (%)? A lab value or soil health card figure is fine.",
     "decides whether carbon-building measures have a big or small payoff"),
    ("climate", ["rainfall_mm", "rainfall_class", "climate"], "What is the rainfall pattern (mm/year, or low/medium/high) and climate zone?",
     "decides whether water-using measures like cover crops help or hurt"),
    ("land_use", ["land_use", "crop_system"], "What is the land used for now (crop and whether it is a monoculture, pasture, forest...)?",
     "decides which interventions are even possible and whether habitat is fragmented"),
    ("land_use", ["native_habitat_pct"], "Roughly what % of the land/landscape is still native vegetation (hedges, grassland, trees)?",
     "tests against the 20% native-habitat minimum"),
    ("human_impact", ["pesticide_use", "pollution"], "How heavy is pesticide or chemical use (low/medium/high)?",
     "chemical pressure can cancel out habitat gains"),
    ("soil", ["ph"], "Do you know the soil pH?", "pH is the strongest single predictor of soil bacterial diversity"),
]

WHY_RE = re.compile(r"^\s*(why|explain|how (?:does|do|will) (?:that|this|it))", re.I)
RESET_RE = re.compile(r"\b(reset|start over|new site|different site)\b", re.I)
QUESTION_RE = re.compile(r"^\s*(what|how|does|do|is|are|can|which|why does)\b.*\?\s*$", re.I)
PERSONAL_RE = re.compile(r"\b(my|our|i have|we have|mine)\b|\d", re.I)


def is_kb_question(msg: str) -> bool:
    """General science question (answer from the KB) vs. a description of the user's own site."""
    return bool(QUESTION_RE.match(msg)) and not PERSONAL_RE.search(msg)


def _questions(p: SiteProfile) -> list[str]:
    groups = p.variable_groups()
    out, used_groups = [], set()
    for g, fields, q, why in QUESTIONS:  # first pass: cover missing families
        if g not in groups and g not in used_groups and all(getattr(p, f) is None for f in fields):
            out.append(f"{q} ({why})")
            used_groups.add(g)
    for _g, fields, q, why in QUESTIONS:  # second pass: most useful remaining fields
        if len(out) >= 3:
            break
        s = f"{q} ({why})"
        if s not in out and all(getattr(p, f) is None for f in fields):
            out.append(s)
    return out[:3]


def _conflicts(p: SiteProfile) -> list[str]:
    w = []
    if p.climate in {"humid", "tropical"} and p.rainfall_mm is not None and p.rainfall_mm < 400:
        w.append(f"Climate '{p.climate}' conflicts with {p.rainfall_mm:.0f} mm rainfall; please confirm one.")
    if p.soc_pct is not None and p.soc_pct > 12:
        w.append(f"SOC {p.soc_pct}% is typical of organic/peat soils, not mineral cropland; confirm the unit (% vs g/kg).")
    return w


def _render(diag, recs, rejected, p: SiteProfile) -> str:
    lines = ["**Diagnosis (variables combined):** " + "; ".join(f"{d['label']} [{', '.join(d['evidence'])}]" for d in diag), ""]
    for i, r in enumerate(recs, 1):
        lines += [f"**{i}. {r.action}**",
                  "Why it works here: " + " ".join(r.reasoning_chain),
                  f"Expected effect: {r.expected_effect}",
                  f"Improves: {', '.join(r.impacted_metrics)} | Horizon: {r.time_horizon} | Confidence: {r.confidence} ({r.confidence_reason})"]
        if r.caveats:
            lines.append("Caveats: " + " ".join(r.caveats))
        lines.append("Evidence: " + "; ".join(f"{c.short} doi:{c.doi}" + (" (verify figure)" if c.verification == "needs-check" else "") for c in r.citations))
        lines.append("")
    for x in rejected:
        lines.append(f"**Deliberately not recommended:** {x.get('name', x['id'])} - {x['why']}")
    return "\n".join(lines).strip()


def _explain(last: list[dict]) -> str:
    if not last:
        return "There is no recommendation yet to explain. Share a few site details first."
    r = last[0]
    return (f"Top recommendation: {r['action']}.\n" + "\n".join(f"- {s}" for s in r["reasoning_chain"]) +
            f"\nEvidence: {', '.join(c['short'] for c in r['citations'])}.")


def _kb_answer(q: str) -> tuple[str, list[dict]]:
    kb, hits = load_kb(), get_retriever().search(q, k=3)
    if not hits:
        return "I could not find anything in the knowledge base on that, so I won't guess.", []
    lines = []
    for h in hits:
        c = kb.evidence.get(h["source_id"])
        src = f"{c['authors']} ({c['year']})" if c else h["chunk_id"]
        lines.append(f"- {c['finding'].strip() if c else h['text']} [{src}]")
    return "From the knowledge base:\n" + "\n".join(lines), hits


def handle(req: ChatRequest) -> ChatResponse:
    kb, retr = load_kb(), get_retriever()
    sess = store.get_or_create(req.session_id)
    sid, warnings = sess["id"], []
    profile = SiteProfile(**sess["profile"]) if sess["profile"] else SiteProfile()
    msg = (req.message or "").strip()
    if msg:
        store.log(sid, "user", msg)

    if msg and RESET_RE.search(msg):
        profile = SiteProfile()
        store.save(sid, {}, [], [])

    if msg and WHY_RE.match(msg) and sess["last_recs"]:
        text = _explain(sess["last_recs"])
        store.log(sid, "assistant", text)
        return ChatResponse(session_id=sid, kind="explanation", text=text, profile=profile)

    if msg and not req.site and is_kb_question(msg):
        text, hits = _kb_answer(msg)
        store.log(sid, "assistant", text)
        return ChatResponse(session_id=sid, kind="answer", text=text, profile=profile, retrieval_trace=hits)

    try:
        new = SiteProfile()
        if req.site:
            new = new.merge(parse_structured(req.site))
        if msg:
            parsed = parse_text(msg)
            extra = {k: v for k, v in llm.extract(msg).items() if getattr(parsed, k, None) is None and k in SiteProfile.model_fields}
            if extra:
                try:
                    parsed = parsed.merge(SiteProfile(**extra, provenance={k: "user" for k in extra}))
                except ValidationError:
                    warnings.append("Ignored an LLM-extracted value that failed validation.")
            new = new.merge(parsed)
    except ValidationError as e:
        bad = ", ".join(str(err["loc"][0]) for err in e.errors())
        text = f"Some values look out of range or invalid ({bad}). Could you re-check them?"
        return ChatResponse(session_id=sid, kind="clarification", text=text, profile=profile, questions=[text])

    profile = profile.merge(new)
    profile, geo_notes = geo.enrich(profile)
    warnings += geo_notes + _conflicts(profile)

    if len(profile.variable_groups()) < MIN_GROUPS or len(profile.known()) < MIN_GROUPS:
        qs = _questions(profile)
        known = ", ".join(profile.known()) or "nothing yet"
        text = ("To give evidence-based advice I need at least three kinds of variables. "
                f"So far I have: {known}.\n" + "\n".join(f"- {q}" for q in qs))
        store.save(sid, profile.model_dump(), pending=qs)
        store.log(sid, "assistant", text)
        return ChatResponse(session_id=sid, kind="clarification", text=text, profile=profile, questions=qs, warnings=warnings)

    diag, recs, rejected, trace = recommend(profile, kb, retr)
    if not recs:
        text = ("None of the site's measured variables cross a problem threshold in my knowledge base, so I won't "
                "invent a recommendation. Useful next data: " + "; ".join(_questions(profile)))
        store.log(sid, "assistant", text)
        return ChatResponse(session_id=sid, kind="answer", text=text, profile=profile, diagnosis=diag, warnings=warnings)

    rec_dicts = [r.model_dump() for r in recs]
    text = _render(diag, recs, rejected, profile)
    summary, note = llm.summarize(rec_dicts, kb.evidence, profile.model_dump(exclude={"provenance"}))
    if summary:
        text = summary + "\n\n" + text
    if note:
        warnings.append(note)
    store.save(sid, profile.model_dump(), rec_dicts, [])
    store.log(sid, "assistant", text)
    return ChatResponse(session_id=sid, kind="recommendations", text=text, profile=profile, diagnosis=diag,
                        recommendations=recs, retrieval_trace=trace, warnings=warnings,
                        questions=_questions(profile)[:1])
