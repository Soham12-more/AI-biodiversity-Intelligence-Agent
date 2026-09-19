from app.dialogue import handle
from app.llm import grounded
from app.models import ChatRequest


def test_asks_clarifying_questions_then_remembers():
    r1 = handle(ChatRequest(message="Biodiversity is declining on my land"))
    assert r1.kind == "clarification" and len(r1.questions) == 3
    r2 = handle(ChatRequest(session_id=r1.session_id, message="SOC 0.3%, rainfall is low"))
    assert r2.kind == "clarification"  # still only 2 variable families
    r3 = handle(ChatRequest(session_id=r1.session_id, message="monoculture wheat field"))
    assert r3.kind == "recommendations"
    assert r3.profile.soc_pct == 0.3 and r3.profile.rainfall_class == "low"  # memory across turns
    r4 = handle(ChatRequest(session_id=r1.session_id, message="why?"))
    assert r4.kind == "explanation" and "Evidence" in r4.text


def test_structured_json_input():
    r = handle(ChatRequest(site={"soc_pct": 0.3, "rainfall_class": "low", "climate": "semi-arid",
                                 "land_use": "cropland", "crop_system": "monoculture"}))
    assert r.kind == "recommendations" and r.retrieval_trace


def test_general_question_answered_from_kb_not_treated_as_site():
    r = handle(ChatRequest(message="How does habitat fragmentation affect species?"))
    assert r.kind == "answer" and "Haddad" in r.text


def test_grounding_guard_blocks_invented_numbers():
    ok, bad = grounded("Cover crops raise SOC by 25% in 3 years (FAO).", {"27", "22", "2.5"})
    assert not ok and "25" in bad


def test_invalid_values_rejected():
    r = handle(ChatRequest(site={"ph": 19}))
    assert r.kind == "clarification"


def test_claim_check_flags_unsupported_percentage():
    r = handle(ChatRequest(message="Cover crops raise SOC by 25% in 3 years, right?"))
    assert r.kind == "answer"
    assert "doesn't match" in r.text
    assert "25" not in r.text.split("\n")[0]  # verdict line must not echo the false figure back as fact


def test_claim_check_confirms_supported_percentage():
    r = handle(ChatRequest(message="Cover crops raise soil microbial abundance by 27%, is that true?"))
    assert r.kind == "answer" and "consistent" in r.text and "27%" in r.text


def test_claim_check_does_not_swallow_number_as_site_data():
    """A claim with digits ('3 years', '25%') must not silently become site variables and trigger
    a full recommendation run instead of answering the question that was actually asked."""
    r1 = handle(ChatRequest(message="SOC is 0.3%, rainfall is low, monoculture wheat, semi-arid region"))
    r2 = handle(ChatRequest(session_id=r1.session_id, message="Cover crops raise SOC by 25% in 3 years, right?"))
    assert r2.kind == "answer"
    assert r2.profile.soc_pct == 0.3  # existing session data untouched, not overwritten by the claim's numbers


def test_bare_factual_question_with_number_not_treated_as_site_data():
    r = handle(ChatRequest(message="Does intercropping have an LER of 1.30?"))
    assert r.kind == "answer" and "1.30" in r.text


def test_claim_check_ignores_unrelated_percentage_from_a_different_card():
    """A claimed % must only be checked against the topically relevant (top-ranked) card, not any
    card in the top-k that happens to share a digit on an unrelated topic (here: LER vs habitat %)."""
    r = handle(ChatRequest(message="Intercropping gives a 50% land equivalent ratio boost, correct?"))
    assert r.kind == "answer" and "doesn't match" in r.text


def test_claim_check_recognizes_is_that_accurate_phrasing():
    r = handle(ChatRequest(message="Fragmentation reduces biodiversity by up to 75%, is that accurate?"))
    assert r.kind == "answer" and "consistent" in r.text
