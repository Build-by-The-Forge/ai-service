"""The evidence rule, portable: an answer is grounded only if it cites at least
one passage it was given and cites nothing else; otherwise it is NOT_COVERED.

The same rule as ``app/services/rag/evidence.py`` (chat and question
generation), restated here so a device runtime can apply it without importing
the server's retrieval stack. A test keeps the abstention text equal to the
server's.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.core.agentic.ports import EvidencePassage

NOT_COVERED = (
    "Your course materials don't cover this yet, so I can't answer it from them. "
    "Try rephrasing, or ask your lecturer to upload material on this topic."
)
_CITATION = re.compile(r"\[(\d{1,2})\]")


def prompt_block(passages: Iterable[EvidencePassage]) -> str:
    """Numbered passages for a prompt: '[n] (document, pages) text'."""
    return "\n\n".join(f"[{p.number}] ({p.document_id}, {_pages(p)}) {p.text}" for p in passages)


def _pages(p: EvidencePassage) -> str:
    if p.page_start is None:
        return "page unknown"
    if p.page_end in (None, p.page_start):
        return f"p. {p.page_start}"
    return f"pp. {p.page_start}-{p.page_end}"


def cited_numbers(answer: str) -> set[int]:
    return {int(n) for n in _CITATION.findall(answer)}


def check_grounded(
    answer: str, passages: tuple[EvidencePassage, ...]
) -> tuple[bool, tuple[EvidencePassage, ...]]:
    """(grounded, the passages cited). Not grounded when nothing was given,
    nothing is cited, or anything cited is not one of the passages."""
    cited = cited_numbers(answer)
    known = {p.number: p for p in passages}
    if not passages or not cited or not cited <= set(known):
        return False, ()
    return True, tuple(known[n] for n in sorted(cited))
