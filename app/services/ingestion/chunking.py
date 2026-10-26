"""Page-aware chunking of an extracted document (P3-3).

Each page's normalized text is split into paragraphs (blank lines), and
paragraphs are packed in order into chunks of at most ``max_chars``. A
paragraph longer than that is split at sentence ends, and a sentence longer
than that at whitespace. Each chunk after the first starts with the end of
the previous one, about ``overlap_chars`` of it, in whole sentences, so a
fact cut at a boundary survives in one chunk.

Every chunk records the pages its text came from (``page_start`` to
``page_end``) for citations, and has a deterministic id derived from the
tenant, asset, run and position: the same input gives the same chunks and
ids, so re-indexing replaces, never duplicates.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.services.ingestion.extraction import ExtractedPage
from app.services.ingestion.normalize import normalize_text

CHUNK_NAMESPACE = uuid.UUID("2d6f4a1e-8c3b-4f7d-9e15-6a0b3c8d2f41")
_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    index: int
    text: str
    page_start: int
    page_end: int


@dataclass(frozen=True, slots=True)
class _Piece:
    text: str
    page: int
    paragraph: int  # pieces of one paragraph re-join with a space, not a break


def _join(pieces: Sequence[_Piece]) -> str:
    parts: list[str] = []
    for i, piece in enumerate(pieces):
        if i:
            parts.append(" " if piece.paragraph == pieces[i - 1].paragraph else "\n\n")
        parts.append(piece.text)
    return "".join(parts)


def chunk_id(tenant_id: str, asset_id: str, run_number: int, index: int) -> str:
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{tenant_id}:{asset_id}:{run_number}:{index}"))


def _split_long(text: str, max_chars: int) -> list[str]:
    """Sentences, and any sentence over max_chars at whitespace."""
    out: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            out.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            out.append(sentence)
    return out


def _pieces(pages: Iterable[ExtractedPage], max_chars: int) -> list[_Piece]:
    pieces: list[_Piece] = []
    number = 0
    for page in pages:
        text = normalize_text(page.text)
        for paragraph in (p.strip() for p in text.split("\n\n")):
            if not paragraph:
                continue
            number += 1
            if len(paragraph) <= max_chars:
                pieces.append(_Piece(paragraph, page.number, number))
            else:
                pieces.extend(
                    _Piece(part, page.number, number) for part in _split_long(paragraph, max_chars)
                )
    return pieces


def _tail(pieces: Sequence[_Piece], overlap_chars: int) -> list[_Piece]:
    """The last whole sentences of a chunk, about overlap_chars of them."""
    if overlap_chars <= 0 or not pieces:
        return []
    last = pieces[-1]
    sentences = [
        _Piece(s, last.page, last.paragraph) for s in _SENTENCE_END.split(last.text) if s.strip()
    ]
    tail: list[_Piece] = []
    size = 0
    for sentence in reversed(sentences):
        if tail and size + len(sentence.text) > overlap_chars:
            break
        tail.insert(0, sentence)
        size += len(sentence.text) + 1
    # Never repeat the whole chunk: that would add nothing new.
    return tail if size < sum(len(p.text) for p in pieces) else []


def chunk_document(
    pages: Iterable[ExtractedPage],
    *,
    tenant_id: str,
    asset_id: str,
    run_number: int,
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[Chunk]:
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be at least 0 and below max_chars")
    chunks: list[Chunk] = []
    current: list[_Piece] = []
    carried = 0  # leading pieces of `current` repeated from the previous chunk

    def size() -> int:
        return sum(len(p.text) + 2 for p in current)

    def flush() -> None:
        nonlocal current, carried
        index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=chunk_id(tenant_id, asset_id, run_number, index),
                index=index,
                text=_join(current),
                page_start=min(p.page for p in current),
                page_end=max(p.page for p in current),
            )
        )
        current = _tail(current, overlap_chars)
        carried = len(current)

    for piece in _pieces(pages, max_chars):
        if len(current) > carried and size() + len(piece.text) > max_chars:
            flush()
        if size() + len(piece.text) > max_chars:
            # The carried overlap leaves no room for this piece: drop it.
            current, carried = [], 0
        current.append(piece)
    if len(current) > carried:  # something beyond the carried overlap
        flush()
    return chunks
