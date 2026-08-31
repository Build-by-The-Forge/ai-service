import re
from typing import Any

from app.core.rag.contracts import ChunkProvenance, DocumentChunk


class Chunker:
    """
    Sentence-aware sliding window document chunker.
    Splits continuous text into overlapping semantic passages while preserving metadata and provenance.
    """

    def __init__(self, chunk_size: int = 150, chunk_overlap: int = 30):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        """Backward-compatible alias for chunk_text."""
        return self.chunk_text(text)

    def _split_into_sentences(self, text: str) -> list[str]:
        # Split by paragraph and sentence punctuation
        sentences = re.split(r"(?<=[.!?])\s+|\n\n+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        words = text.strip().split()
        if len(words) <= self.chunk_size:
            return [" ".join(words)]

        chunks: list[str] = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            window = words[i : i + self.chunk_size]
            if not window:
                break
            chunks.append(" ".join(window))
            if i + self.chunk_size >= len(words):
                break

        return chunks

    def chunk_document(
        self,
        tenant_id: str,
        document_id: str,
        document_title: str,
        text: str,
        course_code: str | None = None,
        topic: str | None = None,
        metadata: dict[str, Any] | None = None,
        provenance: ChunkProvenance | None = None,
    ) -> list[DocumentChunk]:
        raw_chunks = self.chunk_text(text)
        document_chunks: list[DocumentChunk] = []

        for idx, chunk_str in enumerate(raw_chunks, start=1):
            chunk_meta = dict(metadata or {})
            chunk_meta["chunk_index"] = idx
            chunk_meta["total_chunks"] = len(raw_chunks)

            doc_chunk = DocumentChunk.create(
                tenant_id=tenant_id,
                document_id=document_id,
                document_title=document_title,
                content=chunk_str,
                course_code=course_code,
                topic=topic,
                metadata=chunk_meta,
                provenance=provenance,
            )
            document_chunks.append(doc_chunk)

        return document_chunks
