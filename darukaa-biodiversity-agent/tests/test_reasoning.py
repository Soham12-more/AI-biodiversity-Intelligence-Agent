import re

from app.knowledge import get_retriever, load_kb
from app.models import SiteProfile
from app.reasoning import recommend

KB = load_kb()
R = get_retriever()


def run(**kw):
    return recommend(SiteProfile(**kw), KB, R)


def test_semi_arid_example_rejects_cover_crop_and_suggests_agroforestry_intercrop():
    _, recs, rejected, _ = run(soc_pct=0.3, rainfall_class="low", climate="semi-arid",
                               land_use="cropland", crop="wheat", crop_system="monoculture")
    ids = [r.id for r in recs]
    assert "agroforestry_alley" in ids and "legume_intercrop" in ids
    assert "cover_crop" not in ids
    assert any(x["id"] == "cover_crop" for x in rejected)  # water trap caught


def test_every_recommendation_is_multi_metric_and_cited():
    _, recs, _, _ = run(soc_pct=0.3, rainfall_class="low", climate="semi-arid", land_use="cropland", crop_system="monoculture")
    for r in recs:
        assert r.citations and r.impacted_metrics and r.time_horizon in {"short", "medium", "long"}
        assert r.confidence in {"low", "medium", "high"}
    assert any(len(r.addresses) >= 2 for r in recs)


def test_cover_crop_not_rejected_when_water_is_not_limiting():
    _, _, rejected, _ = run(soc_pct=0.6, rainfall_mm=1100, climate="humid", land_use="cropland", crop_system="monoculture")
    assert not any(x["id"] == "cover_crop" for x in rejected)


def test_no_agroforestry_on_forest_land():
    _, recs, _, _ = run(soc_pct=0.8, land_use="forest", fragmentation="high", rainfall_class="medium")
    ids = [r.id for r in recs]
    assert "agroforestry_alley" not in ids and "corridor_restoration" in ids


def test_acidic_soil_triggers_ph_correction():
    _, recs, _, _ = run(ph=4.8, soc_pct=1.5, land_use="cropland", rainfall_mm=900)
    assert "ph_correction" in [r.id for r in recs]


def test_estimated_data_lowers_confidence():
    p = SiteProfile(soc_pct=0.3, rainfall_mm=350, land_use="cropland", crop_system="monoculture",
                    provenance={"soc_pct": "soilgrids", "rainfall_mm": "open-meteo"})
    _, recs, _, _ = recommend(p, KB, R)
    assert all(r.confidence != "high" for r in recs)


def test_kb_integrity_every_number_in_effects_is_in_cited_cards():
    num = re.compile(r"\d+(?:\.\d+)?")
    for iv in KB.interventions:
        allowed = set()
        for e in iv["evidence"] + iv.get("caveat_evidence", []):
            allowed |= set(num.findall(KB.evidence[e]["finding"])) | {str(KB.evidence[e]["year"])}
        for n in num.findall(iv["effect"]) + num.findall(iv.get("caveat", "")):
            assert n in allowed, f"{iv['id']}: number {n} is not in any cited evidence card"
