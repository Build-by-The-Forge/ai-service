"""Text normalization for indexing (P3-3, closes audit F2).

The old TextCleaner deleted every character outside letters, digits and a
few punctuation marks, destroying mathematics (², ≤, π, ∫, =, +) and
Greek. This keeps every printable character:

- Unicode NFC, never NFKC: NFKC would turn ² into 2 and ½ into 1⁄2;
- control characters other than newline and tab are removed;
- runs of spaces and tabs inside a line become one space, trailing
  spaces go, and three or more line breaks become a paragraph break;
- a word split by a hyphen at a line end ("differen-\\ntiation") is joined
  when both sides are lowercase letters.
"""

from __future__ import annotations

import re
import unicodedata

_SPACES = re.compile(r"[ \t  -   　]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(?<=[a-z])-\n(?=[a-z])")


def _printable(ch: str) -> bool:
    return ch in "\n\t" or unicodedata.category(ch)[0] != "C"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if _printable(ch))
    text = "\n".join(_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    text = _HYPHEN_BREAK.sub("", text)
    return _BLANK_LINES.sub("\n\n", text).strip()
