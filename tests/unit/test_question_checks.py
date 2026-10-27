"""The deterministic support checks for generated questions, measured on
the labelled sample set (data/question_checks/v1)."""

import json
from pathlib import Path

from app.services.generation.question_checks import (
    NOT_ANSWERABLE,
    UNSUPPORTED_PREMISE,
    UNTRACEABLE,
    check_question,
    count_reasons,
)
from app.services.rag.evidence import Evidence, Passage

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = json.loads((ROOT / "data/question_checks/v1/samples.json").read_text(encoding="utf-8"))
CORPUS = {
    d["id"]: d
    for d in json.loads((ROOT / "data/retrieval_eval/v1/corpus.json").read_text(encoding="utf-8"))[
        "documents"
    ]
}


def evidence_for(sample) -> Evidence:
    passages = tuple(
        Passage(
            number=n,
            text=CORPUS[p["document"]]["pages"][p["page"] - 1],
            chunk_id=f"{p['document']}-{p['page']}",
            document_id=p["document"],
            page_start=p["page"],
            page_end=p["page"],
        )
        for n, p in enumerate(sample["passages"], start=1)
    )
    return Evidence(query="", passages=passages)


def verdicts():
    return [(s, check_question(s["item"], evidence_for(s))) for s in SAMPLES["samples"]]


def test_every_labelled_sample_gets_its_expected_verdict():
    wrong = [(s["id"], s["expected"], v.reason) for s, v in verdicts() if v.reason != s["expected"]]
    assert wrong == []


def test_rejections_have_full_precision_and_recall_on_the_sample():
    # Rejecting a bad question is the positive class.
    results = [(s["expected"] is not None, not v.ok) for s, v in verdicts()]
    true_pos = sum(bad and rejected for bad, rejected in results)
    precision = true_pos / max(1, sum(rejected for _, rejected in results))
    recall = true_pos / max(1, sum(bad for bad, _ in results))
    assert (precision, recall) == (1.0, 1.0)


def test_the_sample_covers_every_reason_and_supported_questions():
    expected = [s["expected"] for s in SAMPLES["samples"]]
    assert {UNTRACEABLE, UNSUPPORTED_PREMISE, NOT_ANSWERABLE} <= set(expected)
    assert expected.count(None) >= 10 and len(expected) - expected.count(None) >= 10


def test_reasons_are_counted():
    counts = count_reasons(v for _, v in verdicts())
    assert counts == {UNTRACEABLE: 3, UNSUPPORTED_PREMISE: 6, NOT_ANSWERABLE: 5}
