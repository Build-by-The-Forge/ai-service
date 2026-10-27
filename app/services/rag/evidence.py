"""The evidence boundary for document-grounded answers (P3-5).

Chat, explanations and question generation take their course material from
here, so they share one rule for what counts as evidence:

- the passages are the verified tenant's own, retrieved from the durable
  index, in full (not a snippet), numbered [1]..[n] with their document and
  page span;
- an answer is grounded only if it cites at least one of those passages
  and cites nothing else; otherwise, or when nothing was retrieved, the
  answer is NOT_COVERED, never an ungrounded claim.

Retrieval scores cannot tell an answerable question from one the material
does not cover (measured on the evaluation set: the score ranges overlap),
so sufficiency is decided by the citation, not by a score threshold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.rag.contracts import RetrievalFilter, RetrievalPolicy
from app.services.rag.retrieval import HybridRetrievalEngine, get_retrieval_engine

NOT_COVERED = (
    "Your course materials don't cover this yet, so I can't answer it from them. "
    "Try rephrasing, or ask your lecturer to upload material on this topic."
)
PASSAGES = 5
_CITATION = re.compile(r"\[(\d{1,2})\]")


@dataclass(frozen=True)
class Passage:
    number: int
    text: str
    chunk_id: str
    document_id: str
    page_start: int | None
    page_end: int | None

    def citation(self) -> dict:
        return {
            "number": self.number,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


@dataclass(frozen=True)
class Evidence:
    query: str
    passages: tuple[Passage, ...]

    def prompt_block(self) -> str:
        """Numbered passages for a prompt: '[n] (document, pages) text'."""
        return "\n\n".join(
            f"[{p.number}] ({p.document_id}, {_pages(p)}) {p.text}" for p in self.passages
        )

    def context(self) -> list[dict]:
        """The passages as the API's context items (each has `text`)."""
        return [{"text": p.text, **p.citation()} for p in self.passages]


@dataclass(frozen=True)
class GroundedAnswer:
    text: str
    grounded: bool
    citations: list[dict]


def _pages(passage: Passage) -> str:
    if passage.page_start is None:
        return "page unknown"
    if passage.page_end in (None, passage.page_start):
        return f"p. {passage.page_start}"
    return f"pp. {passage.page_start}-{passage.page_end}"


async def gather(
    query: str, tenant_id: str, engine: HybridRetrievalEngine | None = None
) -> Evidence:
    """The tenant's top passages for a query, in full and numbered."""
    engine = engine or get_retrieval_engine()
    context = await engine.retrieve(
        query,
        filter=RetrievalFilter(tenant_id=tenant_id),
        policy=RetrievalPolicy(top_k=PASSAGES),
    )
    passages = []
    for number, retrieved in enumerate(context.chunks, start=1):
        chunk = retrieved.chunk
        meta = chunk.metadata or {}
        page = chunk.provenance.page_number if chunk.provenance else None
        passages.append(
            Passage(
                number=number,
                text=chunk.content,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                page_start=meta.get("page_start", page),
                page_end=meta.get("page_end", page),
            )
        )
    return Evidence(query=query, passages=tuple(passages))


def check_grounded(answer: str, evidence: Evidence) -> GroundedAnswer:
    """Keep an answer only if it cites the evidence it was given, and only that."""
    cited = {int(n) for n in _CITATION.findall(answer)}
    known = {p.number: p for p in evidence.passages}
    if not evidence.passages or not cited or not cited <= set(known):
        return GroundedAnswer(NOT_COVERED, grounded=False, citations=[])
    return GroundedAnswer(
        answer, grounded=True, citations=[known[n].citation() for n in sorted(cited)]
    )
