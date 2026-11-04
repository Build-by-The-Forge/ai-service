"""What retrieval actually supplied, recorded by the system (M3-D provenance).

A recorder belongs to one retrieval. The engine records every chunk the
store returns, after the scope and denial filters and before fusion,
ranking or truncation: a passage retrieved and then cut for length is still
recorded, since leaving it out would rely on knowing what the model didn't
use. Context assembly takes passages only from the recorder, so no text
reaches a model without being recorded. ``sources_used`` is built from the
recorder alone: never from the model's answer or its citations, which the
model controls.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.core.rag.contracts import DocumentChunk


class UnrecordedPassageError(Exception):
    """A passage was offered for a prompt that retrieval didn't record."""


@dataclass(frozen=True, order=True)
class SourceUsed:
    """One document version that reached the model: the asset, and the run
    and attempt whose chunks were retrieved."""

    asset_id: str
    run_number: int
    attempt: int

    def as_dict(self) -> dict[str, str | int]:
        return {"asset_id": self.asset_id, "run_number": self.run_number, "attempt": self.attempt}


def source_of(chunk: DocumentChunk) -> SourceUsed:
    meta = chunk.metadata or {}
    # The durable index always sets both; the in-memory test store indexes
    # one run, written once.
    return SourceUsed(
        chunk.document_id, int(meta.get("run_number", 1)), int(meta.get("attempt", 1))
    )


class RetrievalRecorder:
    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}

    def record(self, chunks: Iterable[DocumentChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def passage(self, chunk_id: str) -> DocumentChunk:
        """The recorded chunk, or UnrecordedPassageError: a prompt may use
        nothing retrieval didn't record."""
        try:
            return self._chunks[chunk_id]
        except KeyError:
            raise UnrecordedPassageError(chunk_id) from None

    def sources_used(self) -> tuple[SourceUsed, ...]:
        return tuple(sorted({source_of(chunk) for chunk in self._chunks.values()}))
