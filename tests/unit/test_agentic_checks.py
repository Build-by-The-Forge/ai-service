"""The foundation checks: never invent a source, a grade or a mastery score."""

import pytest

from app.core.agentic.checks import Ledger, check_final, figures_in, numbers_in
from app.core.agentic.contracts import Boundary, EvidenceRequirements, ToolContract
from app.core.agentic.grounding import NOT_COVERED

IN = {"type": "object", "properties": {}, "additionalProperties": False}
OUT = {"type": "object"}


def evidence_contract():
    return ToolContract(
        name="search_tool",
        version="1.0.0",
        description="Retrieves passages for a query.",
        input_schema=IN,
        output_schema=OUT,
        boundaries=(Boundary.RETRIEVAL,),
        produces_evidence=True,
    )


def plain_contract():
    return ToolContract(
        name="plain_tool",
        version="1.0.0",
        description="Returns passages but is not evidence.",
        input_schema=IN,
        output_schema=OUT,
    )


def passage(chunk, doc="d1", run=1):
    return {"chunk_id": chunk, "document_id": doc, "run_number": run, "attempt": 1}


def ledger(*items):
    led = Ledger()
    led.add_from_output(evidence_contract(), {"passages": list(items)})
    return led


REQ = EvidenceRequirements()


# --- the ledger -----------------------------------------------------------------


def test_only_evidence_tools_fill_the_ledger_and_each_chunk_once():
    led = Ledger()
    assert led.add_from_output(plain_contract(), {"passages": [passage("c1")]}) == 0
    assert (
        led.add_from_output(evidence_contract(), {"passages": [passage("c1"), passage("c2")]}) == 2
    )
    assert (
        led.add_from_output(evidence_contract(), {"citations": [passage("c2"), passage("c3")]}) == 1
    )
    assert led.has("c3") and not led.has("c9")


def test_malformed_passages_are_not_recorded():
    led = Ledger()
    bad = {"passages": [{"no_chunk": 1}, "text", {"chunk_id": 7}, passage("ok")]}
    assert led.add_from_output(evidence_contract(), bad) == 1


# --- rule 1: no invented source -----------------------------------------------------


def test_citing_a_chunk_retrieval_did_not_return_is_a_violation():
    verdict = check_final("Heat moves.", ["c1", "made-up"], ledger(passage("c1")), set(), REQ)
    assert (verdict.verdict, verdict.code) == ("violation", "INVENTED_SOURCE")


def test_an_invented_source_is_a_violation_even_with_an_empty_ledger():
    verdict = check_final("Heat moves.", ["c1"], Ledger(), set(), REQ)
    assert verdict.code == "INVENTED_SOURCE"


# --- rule 2: abstain ------------------------------------------------------------------


def test_an_answer_with_no_citation_abstains_when_sources_are_required():
    verdict = check_final("Heat moves.", [], ledger(passage("c1")), set(), REQ)
    assert (verdict.verdict, verdict.answer) == ("abstain", NOT_COVERED)


def test_no_evidence_at_all_abstains():
    assert check_final("I think so.", [], Ledger(), set(), REQ).verdict == "abstain"


def test_the_minimum_number_of_sources_counts_documents_not_chunks():
    led = ledger(passage("c1", "d1"), passage("c2", "d1"), passage("c3", "d2"))
    req = EvidenceRequirements(min_sources=2)
    assert check_final("x", ["c1", "c2"], led, set(), req).verdict == "abstain"
    assert check_final("x", ["c1", "c3"], led, set(), req).verdict == "ok"


def test_two_versions_of_one_document_are_two_sources():
    led = ledger(passage("c1", "d1", run=1), passage("c2", "d1", run=2))
    assert (
        check_final("x", ["c1", "c2"], led, set(), EvidenceRequirements(min_sources=2)).verdict
        == "ok"
    )


def test_a_task_that_will_not_abstain_fails_instead():
    req = EvidenceRequirements(abstain_if_insufficient=False)
    verdict = check_final("x", [], ledger(passage("c1")), set(), req)
    assert (verdict.verdict, verdict.code) == ("violation", "INSUFFICIENT_EVIDENCE")


def test_a_task_that_needs_no_sources_passes_without_them():
    req = EvidenceRequirements(require_sources=False, min_sources=0)
    verdict = check_final("Here is your plan.", [], Ledger(), set(), req)
    assert verdict.verdict == "ok" and verdict.cited == ()


def test_a_grounded_answer_passes_with_its_citations_once_each():
    verdict = check_final("Heat moves.", ["c1", "c1"], ledger(passage("c1")), set(), REQ)
    assert (verdict.verdict, verdict.cited, verdict.answer) == ("ok", ("c1",), "Heat moves.")


# --- rule 3: no invented grade or mastery --------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "You scored 85% on this topic.",
        "Your mastery is 72 percent, so 72%.",
        "Your score: 17 out of 20.",
        "Mastery of heat transfer is 0.9.",
        "You got 8/10.",
        "Your mark was 65.",
        "Your grade is A.",
        "You were graded B+ for this.",
    ],
)
def test_a_score_no_tool_computed_is_a_violation(text):
    verdict = check_final(text, ["c1"], ledger(passage("c1")), set(), REQ)
    assert (verdict.verdict, verdict.code) == ("violation", "UNSUPPORTED_FIGURE")


def test_a_figure_a_tool_computed_may_be_quoted_as_given_or_as_a_percentage():
    led = ledger(passage("c1"))
    produced = numbers_in({"gaps": [{"topic_id": "heat", "weakness": 0.537, "evidence_count": 3}]})
    assert check_final("Weakness is 0.537 in heat.", ["c1"], led, produced, REQ).verdict == "ok"
    assert check_final("About 53.7% weak in heat.", ["c1"], led, produced, REQ).verdict == "ok"
    assert check_final("About 60% weak in heat.", ["c1"], led, produced, REQ).verdict == "violation"


def test_numbers_that_are_not_scores_are_not_flagged():
    text = "Read pages 12 to 15 and attempt 3 questions on day 2 of the plan."
    assert figures_in(text) == (set(), False)
    assert check_final(text, ["c1"], ledger(passage("c1")), set(), REQ).verdict == "ok"


def test_numbers_in_ignores_booleans_and_finds_nested_numbers():
    assert numbers_in({"a": [1, {"b": 2.5}], "c": True, "d": "7"}) == {1.0, 2.5}


def test_an_invented_source_is_reported_before_an_unsupported_figure():
    verdict = check_final("You scored 99%.", ["ghost"], Ledger(), set(), REQ)
    assert verdict.code == "INVENTED_SOURCE"
