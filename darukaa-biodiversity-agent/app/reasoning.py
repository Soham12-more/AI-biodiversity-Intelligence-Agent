"""Multi-metric reasoning: diagnose stressors from the site profile, rank interventions by how many
co-occurring stressors they relieve, explain the causal chain, and grade confidence.

Design rule: numbers in the output come only from evidence cards or from the user's own data."""
from __future__ import annotations

from .knowledge import KB, Retriever
from .models import Citation, Recommendation, SiteProfile

HIGH_GRADE = {"meta-analysis", "global assessment"}


def _check(profile: SiteProfile, cond: dict) -> bool:
    if cond.get("only_if_missing") and getattr(profile, cond["only_if_missing"]) is not None:
        return False
    v = getattr(profile, cond["field"])
    if v is None:
        return False
    op, target = cond["op"], cond["value"]
    return {"<": lambda: v < target, ">": lambda: v > target, ">=": lambda: v >= target,
            "==": lambda: v == target, "in": lambda: v in target}[op]()


def diagnose(profile: SiteProfile, kb: KB) -> list[dict]:
    out = []
    for sid, s in kb.stressors.items():
        conds = s.get("any") or [s["rule"]]
        hits = [c for c in conds if _check(profile, c)]
        if not hits:
            continue
        sev = 0.7
        if "severity_from" in s:
            sf = s["severity_from"]
            v = getattr(profile, sf["field"])
            sev = max(0.3, min(1.0, (sf["best"] - v) / (sf["best"] - sf["worst"])))
        elif s.get("any"):
            sev = min(1.0, 0.6 + 0.15 * len(hits))  # more independent signals -> more certain
        out.append({"id": sid, "label": s["label"], "severity": round(sev, 2),
                    "evidence": [f"{h['field']}={getattr(profile, h['field'])}" for h in hits],
                    "source": s["source"]})
    return sorted(out, key=lambda x: -x["severity"])


def _site_facts(p: SiteProfile, active: set[str]) -> dict[str, str]:
    """One sentence per active stressor, grounded in the user's numbers and a cited threshold."""
    f = {}
    if "low_soc" in active:
        f["low_soc"] = (f"SOC is {p.soc_pct}%, far below the ~2% level where crop-yield gains level off "
                        f"[oldfield2019], so each unit of added carbon has a high marginal return here.")
    if "water_limited" in active:
        bits = [x for x in [p.rainfall_mm and f"{p.rainfall_mm:.0f} mm rainfall",
                            p.rainfall_class and f"{p.rainfall_class} rainfall", p.climate,
                            p.soil_moisture_class and f"{p.soil_moisture_class} soil moisture"] if x]
        f["water_limited"] = (f"Water is the binding constraint ({', '.join(bits)}), so every option is judged by "
                              f"its water cost as well as its benefit; moisture also decides which species can persist.")
    if "monoculture" in active:
        f["monoculture"] = (f"{(p.crop or 'Single-crop').capitalize()} monoculture offers one habitat type "
                            f"and one food resource for the whole season.")
    if "low_native_habitat" in active:
        f["low_native_habitat"] = (
            f"Native habitat is {p.native_habitat_pct:.0f}% of the land, below the 20% minimum argued for working landscapes [garibaldi2021]."
            if p.native_habitat_pct is not None else
            "With a monoculture and no reported native habitat, the landscape is very likely below the 20% native-habitat minimum [garibaldi2021].")
    if "fragmentation" in active:
        f["fragmentation"] = "Remaining habitat is fragmented; isolated patches keep losing species over time [haddad2015]."
    if "acidic_soil" in active:
        f["acidic_soil"] = f"pH {p.ph} is acidic; soil bacterial diversity is lowest in acidic soils [fierer2006]."
    if "pesticide_pressure" in active:
        f["pesticide_pressure"] = "Chemical pressure removes predators and pollinators that other measures try to bring back [geiger2010]."
    if "heat_stress" in active:
        f["heat_stress"] = f"Mean temperature {p.mean_temp_c} °C raises evaporation and heat stress on soil life."
    if "water_body_exposure" in active:
        f["water_body_exposure"] = "Runoff from the field reaches a water body, linking farm inputs to aquatic species."
    return f


def _cite(kb: KB, eid: str, score: float | None = None) -> Citation:
    c = kb.evidence[eid]
    return Citation(id=eid, short=f"{c['authors']} ({c['year']}), {c['venue']}", doi=c.get("doi"),
                    verification=c["verification"], retrieval_score=score)


def _confidence(iv: dict, kb: KB, p: SiteProfile, active: set[str]) -> tuple[str, str]:
    grades = [kb.evidence[e]["type"] for e in iv["evidence"]]
    level = 2 if all(g in HIGH_GRADE for g in grades) else 1
    reasons = ["meta-analytic evidence" if level == 2 else "evidence is a synthesis/review, not a pooled effect size"]
    if p.land_use is None:
        level -= 1
        reasons.append("land use not confirmed")
    est = [k for k, v in p.provenance.items() if v != "user"]
    if est:
        level -= 1
        reasons.append(f"{', '.join(est)} estimated from global datasets, not measured")
    if "water_limited" in active and iv["id"] in {"agroforestry_alley", "corridor_restoration"}:
        level = min(level, 1)
        reasons.append("global averages; tree establishment in dry sites is the main risk")
    return ["low", "medium", "high"][max(0, min(2, level))], "; ".join(reasons)


def recommend(profile: SiteProfile, kb: KB, retriever: Retriever, top_k: int = 4):
    diag = diagnose(profile, kb)
    active = {d["id"]: d for d in diag}
    facts = _site_facts(profile, set(active))
    recs, rejected, trace = [], [], []

    for iv in kb.interventions:
        if profile.land_use and profile.land_use not in iv["land_use"]:
            continue
        if any(_check(profile, c) for c in iv.get("contraindicated", [])):
            rejected.append({"id": iv["id"], "why": iv.get("caveat", "contraindicated for this land use")})
            continue
        hit = [s for s in iv["addresses"] if s in active]
        if not hit:
            continue
        if iv.get("water_sensitive") and "water_limited" in active:
            rejected.append({"id": iv["id"], "name": iv["name"], "why": iv["caveat"],
                             "evidence": iv.get("caveat_evidence", [])})
            continue

        score = sum(active[s]["severity"] for s in hit) + 0.35 * (len(hit) - 1)  # multi-stressor bonus
        query = f"{iv['name']}. " + " ".join(active[s]["label"] for s in hit) + f" {profile.climate or ''} {profile.crop or ''}"
        allowed = set(iv["evidence"]) | {c.source_id for c in kb.chunks if c.kind == "doc"}
        hits = retriever.search(query, k=4, restrict_to=allowed)
        trace.append({"intervention": iv["id"], "query": query.strip(),
                      "hits": [{k: h[k] for k in ("chunk_id", "score")} for h in hits]})
        best = {h["source_id"]: h["score"] for h in reversed(hits)}

        chain = [facts[s] for s in hit if s in facts]
        chain.append(iv["mechanism"].strip())
        conf, conf_reason = _confidence(iv, kb, profile, set(active))
        caveats = [iv["caveat"]] if iv.get("caveat") else []
        if "water_limited" in active and iv["id"] == "agroforestry_alley":
            caveats.append("Design judgment (not from a cited study): trees also use water, so choose native drought-tolerant species, keep rows wide, and budget establishment watering for the first seasons.")
        score *= {"high": 1.0, "medium": 0.85, "low": 0.7}[conf]
        recs.append(Recommendation(
            id=iv["id"], action=iv["name"], why=iv["mechanism"].strip(), reasoning_chain=chain,
            impacted_metrics=iv["improves"], addresses=hit, expected_effect=iv["effect"],
            time_horizon=iv["horizon"], confidence=conf, confidence_reason=conf_reason, caveats=caveats,
            citations=[_cite(kb, e, best.get(e)) for e in iv["evidence"]], score=round(score, 2)))

    prereq = {iv["id"] for iv in kb.interventions if iv.get("prerequisite")}
    recs.sort(key=lambda r: (r.id not in prereq, -r.score))
    recs = recs[:top_k]
    ids = {r.id for r in recs}
    if {"legume_intercrop", "habitat_setaside"} <= ids:  # the non-obvious pairing: land sparing funds habitat
        for r in recs:
            if r.id == "habitat_setaside":
                r.reasoning_chain.insert(0, "Pairs with intercropping: the land-equivalent gain means strips can be taken out of production without losing total harvest [martinguay2018].")
    return diag, recs, rejected, trace
