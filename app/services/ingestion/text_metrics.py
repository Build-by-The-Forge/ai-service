"""Extraction accuracy against ground truth (P3-2 baseline).

CER: character edit distance / reference length; WER: the same over
whitespace-separated words. Both texts are normalized the same way first
(Unicode NFC, whitespace runs to one space), so line breaks and spacing
do not count, while every character, symbol and word order does.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_distance(a: Sequence[object], b: Sequence[object]) -> int:
    """Levenshtein distance in O(len(a) * len(b)) time, O(len(b)) memory."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item in enumerate(a, start=1):
        current = [i]
        for j, other in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (item != other))
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    return edit_distance(ref, hyp) / max(len(ref), 1)


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    return edit_distance(ref, hyp) / max(len(ref), 1)
