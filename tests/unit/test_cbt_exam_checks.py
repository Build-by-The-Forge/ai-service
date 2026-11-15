"""The CBT exam rules on top of every generated question's checks (AI-10)."""

import pytest

from app.schemas.ai import GeneratedQuestion
from app.services.generation.cbt_exam import (
    MISSING_EXPLANATION,
    NOT_FOUR_OPTION_MCQ,
    UNSUPPORTED_EXPLANATION,
    check_exam_question,
    citations,
)
from app.services.generation.question_checks import supported
from app.services.rag.evidence import Passage
from tests.cbt_support import PASSAGE, PASSAGE_ONE, question

PASSAGES = (PASSAGE_ONE,)


def verdict(**changes):
    return check_exam_question(GeneratedQuestion.model_validate(question(**changes)), PASSAGES)


def test_a_grounded_four_option_question_with_a_supported_explanation_passes():
    assert verdict().ok


@pytest.mark.parametrize(
    "changes",
    [
        {
            "options": [
                {"text": "Nucleus", "is_correct": False},
                {"text": "Mitochondrion", "is_correct": True},
                {"text": "Ribosome", "is_correct": False},
            ]
        },
        {
            "options": [
                {"text": "Nucleus", "is_correct": False},
                {"text": "Mitochondrion", "is_correct": True},
                {"text": "Ribosome", "is_correct": False},
                {"text": "Cell wall", "is_correct": False},
                {"text": "Vacuole", "is_correct": False},
            ]
        },
        {
            "question_type": "true_false",
            "options": [
                {"text": "True", "is_correct": True},
                {"text": "False", "is_correct": False},
            ],
        },
    ],
    ids=["three options", "five options", "true_false"],
)
def test_only_a_four_option_mcq_is_an_exam_question(changes):
    assert verdict(**changes).reason == NOT_FOUR_OPTION_MCQ


@pytest.mark.parametrize("explanation", [None, "", "   "])
def test_an_exam_question_needs_an_explanation(explanation):
    assert verdict(explanation=explanation).reason == MISSING_EXPLANATION


@pytest.mark.parametrize(
    "explanation",
    [
        "Mitochondria were first described in 1857.",  # a number the passage lacks
        "Albert von Kolliker named this structure after observing muscle tissue samples.",
    ],
    ids=["number", "content"],
)
def test_an_explanation_beyond_the_cited_passages_is_unsupported(explanation):
    assert verdict(explanation=explanation).reason == UNSUPPORTED_EXPLANATION


def test_an_explanation_is_checked_only_against_the_passages_the_question_cites():
    lysosomes = "Lysosomes digest worn organelles with hydrolytic enzymes."
    other = Passage(2, lysosomes, "chunk-2", "asset-other", 1, 1)
    explained_by_other = question(explanation=lysosomes)  # cites [1] only
    assert (
        check_exam_question(
            GeneratedQuestion.model_validate(explained_by_other), (PASSAGE_ONE, other)
        ).reason
        == UNSUPPORTED_EXPLANATION
    )


def test_citations_name_the_exact_document_version_and_never_the_text():
    [cited] = citations(GeneratedQuestion.model_validate(question()), PASSAGES)

    assert cited == {
        "number": 1,
        "document_id": "asset-bio",
        "page_start": 3,
        "page_end": 4,
        "run_number": 2,
        "attempt": 1,
    }
    assert PASSAGE not in str(cited)


def test_supported_is_the_premise_rule_for_any_text():
    assert supported("The nucleus holds the DNA.", PASSAGE)
    assert not supported("The nucleus holds 46 chromosomes.", PASSAGE)
