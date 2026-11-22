"""The foundation checks on an agent's final answer.

Three rules, applied by the runtime before any answer leaves it:

1. Never invent a source. An answer may cite only passages (by chunk id) that a
   retrieval tool returned in this run. The ledger holds those.
2. Abstain when the evidence is not enough. A task that requires sources and
   whose answer rests on too few valid ones gets NOT_COVERED, never a claim.
3. Never invent a grade or a mastery score. A figure in the answer that reads as
   a score, grade, mark, mastery or percentage must be a number a tool computed
   in this run. BUILD computes; the model narrates.

Rule 3 is a text check and says so: it finds figures the way a reader would
(a percentage, a number after "score", "mark", "grade" or "mastery", a letter
grade) and cannot find every phrasing. It is a floor under, not a substitute
for, the structural protection that no tool lets a model set a score. Replacing
the free-text answer with a structured one is a decision for the reviewer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from app.core.agentic.contracts import EvidenceRequirements, ToolContract
from app.core.agentic.grounding import NOT_COVERED

_NUMBER = r"\d+(?:\.\d+)?"
_PERCENT = re.compile(rf"({_NUMBER})\s?%")
_NAMED = re.compile(
    rf"\b(?:score[ds]?|marks?|grades?|mastery|mastered|gpa|rating)\b[^.\n\d]{{0,25}}?({_NUMBER})",
    re.IGNORECASE,
)
_OUT_OF = re.compile(rf"({_NUMBER})\s?(?:/|out of)\s?({_NUMBER})", re.IGNORECASE)
_LETTER = re.compile(r"\bgrade(?:d)?(?: of| is| was)?\s+([A-F][+-]?)(?![A-Za-z])")


@dataclass
class Ledger:
    """What retrieval returned in this run: the only things an answer may cite."""

    passages: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, item: dict[str, Any]) -> bool:
        """Record a passage. True if it is new to the ledger."""
        chunk_id = item.get("chunk_id")
        if not isinstance(chunk_id, str) or chunk_id in self.passages:
            return False
        self.passages[chunk_id] = item
        return True

    def add_from_output(self, contract: ToolContract, data: dict[str, Any]) -> int:
        """Record the passages an evidence tool returned; the count that were new."""
        if not contract.produces_evidence:
            return 0
        added = 0
        for key in ("passages", "citations"):
            items = data.get(key)
            if isinstance(items, list):
                added += sum(1 for item in items if isinstance(item, dict) and self.add(item))
        return added

    def has(self, chunk_id: str) -> bool:
        return chunk_id in self.passages

    def sources(self, chunk_ids: list[str]) -> set[tuple[str, int, int]]:
        return {
            (
                self.passages[c].get("document_id", ""),
                int(self.passages[c].get("run_number", 1)),
                int(self.passages[c].get("attempt", 1)),
            )
            for c in chunk_ids
            if c in self.passages
        }


def numbers_in(value: Any) -> set[float]:
    """Every number a tool output holds, however deep."""
    found: set[float] = set()
    if isinstance(value, bool):
        return found
    if isinstance(value, (int, float)):
        found.add(float(value))
    elif isinstance(value, dict):
        for item in value.values():
            found |= numbers_in(item)
    elif isinstance(value, list):
        for item in value:
            found |= numbers_in(item)
    return found


def figures_in(text: str) -> tuple[set[float], bool]:
    """(numbers the text presents as a score, grade or percentage, whether it
    names a letter grade)."""
    figures = {float(m) for m in _PERCENT.findall(text)}
    figures |= {float(m) for m in _NAMED.findall(text)}
    for num, den in _OUT_OF.findall(text):
        figures |= {float(num), float(den)}
    return figures, bool(_LETTER.search(text))


def _supported(figure: float, produced: set[float]) -> bool:
    for n in produced:
        if abs(figure - n) < 1e-9:  # the number as the tool gave it
            return True
        if abs(figure - n * 100.0) < 0.051:  # a 0..1 figure quoted as a percentage
            return True
    return False


Verdict = Literal["ok", "abstain", "violation"]


@dataclass(frozen=True)
class FinalVerdict:
    verdict: Verdict
    code: str | None = None
    # The answer to give: the model's (ok), or NOT_COVERED (abstain).
    answer: str | None = None
    cited: tuple[str, ...] = ()


def check_final(
    answer: str,
    citations: list[str],
    ledger: Ledger,
    produced_numbers: set[float],
    requirements: EvidenceRequirements,
) -> FinalVerdict:
    # 1. A source retrieval did not return is invented.
    if any(not ledger.has(c) for c in citations):
        return FinalVerdict("violation", "INVENTED_SOURCE")
    cited = tuple(dict.fromkeys(citations))
    # 2. Not enough valid evidence: abstain (or fail, if the task would not abstain).
    if requirements.require_sources and len(ledger.sources(list(cited))) < max(
        1, requirements.min_sources
    ):
        if requirements.abstain_if_insufficient:
            return FinalVerdict("abstain", "INSUFFICIENT_EVIDENCE", NOT_COVERED)
        return FinalVerdict("violation", "INSUFFICIENT_EVIDENCE")
    # 3. A grade or score no tool computed.
    figures, letter = figures_in(answer)
    if letter or any(not _supported(f, produced_numbers) for f in figures):
        return FinalVerdict("violation", "UNSUPPORTED_FIGURE")
    return FinalVerdict("ok", None, answer, cited)
