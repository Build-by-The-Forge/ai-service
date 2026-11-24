"""AI-10's topic-coverage checks, held to their measured baseline.

The thresholds are the ones the evaluation chose on the dev split
(data/topic_coverage/v1/baseline.json); the relevance check decides every
labelled pair as the evaluated signal did; the relevance measurement
reproduces the committed baseline exactly (it needs no index, so it runs
here). The regression cases name what the checks must keep doing, and what
they are known not to do.
"""

import json

import pytest

from app.schemas.ai import GeneratedQuestion
from app.services.generation import topic_coverage
from app.services.generation.topic_coverage import OFF_TOPIC, check_relevance, covers
from app.services.rag.evidence import Passage
from tests.unit.test_topic_coverage_dataset import DATASET, evaluate

BASELINE = json.loads((DATASET / "baseline.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dataset():
    return evaluate.load_dataset()


def page(dataset, document: str, number: int, n: int = 1) -> Passage:
    return Passage(n, dataset.pages[document][number - 1], f"c{n}", document, number, number)


def test_the_baseline_was_measured_on_this_exact_dataset(dataset):
    assert BASELINE["dataset"]["sha256"] == dataset.sha256


def test_the_thresholds_are_the_ones_chosen_on_the_dev_split():
    assert BASELINE["coverage"]["selected"] == {
        "signal": "lexical_best_passage",
        "threshold": topic_coverage.TOPIC_COVERAGE,
    }
    assert BASELINE["relevance"]["selected"] == {
        "signal": "cited_passage",
        "threshold": topic_coverage.QUESTION_RELEVANCE,
    }


def test_the_relevance_measurement_reproduces_the_baseline(dataset):
    assert evaluate.evaluate_relevance(dataset) == BASELINE["relevance"]


def test_the_relevance_check_decides_every_pair_as_measured(dataset):
    disagree = []
    for q in dataset.questions:
        passages = [
            page(dataset, p["document"], p["page"], n) for n, p in enumerate(q["passages"], 1)
        ]
        question = GeneratedQuestion.model_validate(q["item"])
        for tid in q["on_topic"] + q["off_topic"]:
            topic = dataset.topics[tid]["topic"]
            signal = evaluate.relevance_signals(topic, q["item"], [p.text for p in passages])
            measured = signal["cited_passage"] >= BASELINE["relevance"]["selected"]["threshold"]
            if check_relevance(topic, question, passages).ok != measured:
                disagree.append((q["id"], tid))
    assert disagree == []


# --- regression cases --------------------------------------------------------------------


@pytest.mark.parametrize(
    "topic,document,number",
    [
        ("Quantum computing", "bio101-cell", 1),  # the #62 review's case
        ("Facilitated diffusion", "bio101-cell", 2),  # shares "diffusion"
        ("Alternating current", "phy102-current", 1),  # shares "current"
        ("Integration by parts", "mts102-differentiation", 2),
    ],
)
def test_an_uncovered_topic_is_not_covered_by_a_passage_sharing_some_of_its_words(
    dataset, topic, document, number
):
    assert not covers(topic, [page(dataset, document, number)])


@pytest.mark.parametrize(
    "topic,document,number",
    [
        ("Osmosis", "bio101-cell", 2),
        ("Coulomb's law", "phy102-electrostatics", 2),
        ("Hess's law", "chm101-thermochemistry", 2),
        ("The chain rule", "mts102-differentiation", 2),
    ],
)
def test_a_topic_the_notes_name_is_covered(dataset, topic, document, number):
    assert covers(topic, [page(dataset, document, 1, n=2), page(dataset, document, number)])


def test_known_limit_a_topic_named_in_other_words_is_not_covered(dataset):
    # "Averages" are taught as the mean, median and mode: a false negative
    # the baseline counts. The check reads words, not meaning.
    assert not covers("Averages", [page(dataset, "sta111-descriptive", 1)])


def test_known_limit_a_near_miss_sharing_most_of_the_notes_words_is_covered(dataset):
    # The notes teach resistors in series and parallel, not capacitors; two
    # of the topic's three words are enough (a measured false positive).
    # Relevance and the support checks still apply after generation.
    assert covers("Capacitors in series and parallel", [page(dataset, "phy102-current", 2)])


def test_a_question_citing_only_another_topics_passage_is_off_topic(dataset):
    question = GeneratedQuestion.model_validate(dataset.questions[0]["item"])  # limits, page 1
    limits = page(dataset, "mts102-limits", 1)
    osmosis = page(dataset, "bio101-cell", 2, n=2)

    assert check_relevance("Osmosis", question, [limits, osmosis]).reason == OFF_TOPIC
    assert check_relevance("Limits of functions", question, [limits, osmosis]).ok


def test_no_topic_terms_means_no_coverage(dataset):
    assert not covers("the", [page(dataset, "bio101-cell", 1)])
