"""What a CBT exam question must be, beyond any generated question (AI-10).

StructuredQuestionGenerator has already checked each question's shape
(exactly one correct option, distinct options) and its support (it cites
passages the model was given, and its premise and correct answer don't go
beyond them). An exam question must also:

- be multiple choice with exactly four options (A to D), as the prompt asks;
- carry an explanation, held to the premise rule: no number, and few
  content terms, beyond the passages it cites.

The explanation reuses the premise threshold. That threshold was measured on
prompts and answers (data/question_checks/v1), not explanations, so for
explanations it is a floor until they are labelled and measured.
"""

from __future__ import annotations

from typing import Any

from app.schemas.ai import GeneratedQuestion
from app.services.generation.question_checks import Verdict, supported
from app.services.rag.evidence import Passage

CBT_OPTIONS = 4

NOT_FOUR_OPTION_MCQ = "not_a_four_option_mcq"
MISSING_EXPLANATION = "missing_explanation"
UNSUPPORTED_EXPLANATION = "unsupported_explanation"


def _cited(question: GeneratedQuestion, passages: tuple[Passage, ...]) -> list[Passage]:
    by_number = {passage.number: passage for passage in passages}
    return [by_number[n] for n in sorted(set(question.sources)) if n in by_number]


def check_exam_question(question: GeneratedQuestion, passages: tuple[Passage, ...]) -> Verdict:
    if question.question_type != "mcq" or len(question.options) != CBT_OPTIONS:
        return Verdict(False, NOT_FOUR_OPTION_MCQ)
    explanation = (question.explanation or "").strip()
    if not explanation:
        return Verdict(False, MISSING_EXPLANATION)
    cited = _cited(question, passages)
    if not cited or not supported(explanation, "\n".join(p.text for p in cited)):
        return Verdict(False, UNSUPPORTED_EXPLANATION)
    return Verdict(True)


def citations(question: GeneratedQuestion, passages: tuple[Passage, ...]) -> list[dict[str, Any]]:
    """The exact document versions a question cites: document, pages, run and
    attempt, from the passages the model was given. Never passage text."""
    return [
        {
            "number": passage.number,
            "document_id": passage.document_id,
            "page_start": passage.page_start,
            "page_end": passage.page_end,
            "run_number": passage.run_number,
            "attempt": passage.attempt,
        }
        for passage in _cited(question, passages)
    ]
