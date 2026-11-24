"""Whether exam material and questions are on the requested topic (AI-10).

Retrieval returns its top passages even for a topic they don't cover, and
its scores can't tell (ADR 001); the support checks keep any question its
cited passages support, whatever it is about. Two lexical, deterministic
checks close that gap for the exam generator:

- **before generation (coverage):** the requested topic's content terms
  must appear in one retrieved passage, at least TOPIC_COVERAGE of them;
  otherwise the material doesn't cover the topic and the model isn't asked;
- **after generation (relevance):** the topic's content terms must appear
  in a passage the question cites, at least QUESTION_RELEVANCE of them;
  otherwise the question is off topic and is discarded.

Both thresholds were chosen on the dev split of a labelled set
(data/topic_coverage/v1) and measured on its held-out test split
(scripts/evaluate_topic_coverage.py; the committed baseline.json records
the false positives and negatives with their intervals). They are words,
not meaning: a topic named only in other words is rejected, and a near miss
that shares the notes' words can pass. They lower the risk of an off-topic
exam; they don't remove it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from app.schemas.ai import GeneratedQuestion
from app.services.generation.question_checks import Verdict, content_terms
from app.services.rag.evidence import Passage

TOPIC_COVERAGE = 0.6
QUESTION_RELEVANCE = 0.4

OFF_TOPIC = "off_topic"


def _best(topic: str, texts: Iterable[str]) -> float:
    terms = content_terms(topic)
    if not terms:
        return 0.0
    return max((len(terms & content_terms(text)) / len(terms) for text in texts), default=0.0)


def covers(topic: str, passages: Sequence[Passage]) -> bool:
    """Whether the retrieved passages cover the topic (before generation)."""
    return _best(topic, (p.text for p in passages)) >= TOPIC_COVERAGE


def check_relevance(
    topic: str, question: GeneratedQuestion, passages: Sequence[Passage]
) -> Verdict:
    """Whether a supported question is on the requested topic (after generation)."""
    cited = set(question.sources)
    if _best(topic, (p.text for p in passages if p.number in cited)) >= QUESTION_RELEVANCE:
        return Verdict(True)
    return Verdict(False, OFF_TOPIC)
