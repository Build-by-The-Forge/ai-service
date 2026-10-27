"""Deterministic support checks for generated questions (P3-5 follow-up).

A generated question is kept only if it is traceable to the course passages
it was generated from and does not go beyond them. These checks are lexical
and deterministic; they do not ask a model to judge another model:

- traceable: the question names its source passages (`sources`), all of
  which were given to the model;
- no unsupported premise: every number in the question appears in its
  sources, and most of its content terms do;
- answerable from the sources (multiple choice): every number in the
  correct option appears in its sources, and most of its content terms do.

The thresholds were set on a labelled sample (data/question_checks/v1),
which reports the checks' precision and recall; lexical checks can still
pass a question that rearranges true terms into a false claim, so they are
a floor, not proof of correctness.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from app.services.rag.evidence import Evidence

PREMISE_COVERAGE = 0.6
ANSWER_COVERAGE = 0.6

UNTRACEABLE = "untraceable"
UNSUPPORTED_PREMISE = "unsupported_premise"
NOT_ANSWERABLE = "not_answerable_from_sources"

_TOKEN = re.compile(r"\d+(?:[.,]\d+)?|[^\W\d_]{3,}")
# Function words and assessment phrasing carry no subject content.
_STOPWORDS = frozenset("""
    the and for are was were but not with that this these those from into onto
    than then them they their there here have has had its who whom whose what
    which when where why how does did doing done can could should would will
    shall may might must about above below over under between among each every
    other another such only also very more most less least much many some any
    all both either neither one two three four five being been you your our
    find calculate state explain define describe give list name identify
    determine evaluate compute show write following statement true false
    correct incorrect answer question best option options none above value
    values given using used use according passage text example examples
    """.split())


def _stem(word: str) -> str:
    for suffix in ("ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def _tokens(text: str) -> list[str]:
    return [t.replace(",", ".") for t in _TOKEN.findall(text.lower())]


def numbers(text: str) -> set[str]:
    return {t for t in _tokens(text) if t[0].isdigit()}


def content_terms(text: str) -> set[str]:
    return {_stem(t) for t in _tokens(text) if not t[0].isdigit() and t not in _STOPWORDS}


def _covered(terms: set[str], available: set[str]) -> float:
    return len(terms & available) / len(terms) if terms else 1.0


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str | None = None


def check_question(
    item: dict[str, Any],
    evidence: Evidence,
    *,
    premise_coverage: float = PREMISE_COVERAGE,
    answer_coverage: float = ANSWER_COVERAGE,
) -> Verdict:
    """Whether one structurally valid generated question is supported by
    the passages it cites."""
    given = {p.number: p.text for p in evidence.passages}
    sources = item.get("sources")
    if (
        not isinstance(sources, list)
        or not sources
        or not all(isinstance(n, int) and n in given for n in sources)
    ):
        return Verdict(False, UNTRACEABLE)
    cited = "\n".join(given[n] for n in sorted(set(sources)))
    available_terms, available_numbers = content_terms(cited), numbers(cited)

    prompt = str(item.get("prompt", ""))
    if not numbers(prompt) <= available_numbers:
        return Verdict(False, UNSUPPORTED_PREMISE)
    if _covered(content_terms(prompt), available_terms) < premise_coverage:
        return Verdict(False, UNSUPPORTED_PREMISE)

    if item.get("question_type") == "mcq":
        correct = [o.get("text", "") for o in item.get("options", []) if o.get("is_correct")]
        answer = " ".join(str(text) for text in correct)
        if not numbers(answer) <= available_numbers:
            return Verdict(False, NOT_ANSWERABLE)
        if _covered(content_terms(answer), available_terms) < answer_coverage:
            return Verdict(False, NOT_ANSWERABLE)
    return Verdict(True)


def count_reasons(verdicts: Iterable[Verdict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for verdict in verdicts:
        if not verdict.ok and verdict.reason:
            counts[verdict.reason] = counts.get(verdict.reason, 0) + 1
    return counts
