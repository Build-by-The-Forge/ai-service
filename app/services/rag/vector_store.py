from abc import ABC, abstractmethod

from app.core.rag.contracts import DocumentChunk, RetrievalFilter
from app.core.rag.errors import TenantContextMissingError, VectorStoreError
from app.services.rag.embedder import cosine_similarity


class VectorStorePort(ABC):
    """
    Abstract Port for vector database storage and similarity search.
    """

    @abstractmethod
    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Insert or update chunks with their dense embeddings."""

    @abstractmethod
    async def query_dense(
        self, vector: list[float], filter: RetrievalFilter, top_k: int = 10
    ) -> list[tuple[DocumentChunk, float]]:
        """Find the top-k nearest semantic chunks adhering to the authoritative tenant filter."""

    @abstractmethod
    async def delete_document(self, tenant_id: str, document_id: str) -> int:
        """Delete all chunks associated with a document under a specific tenant."""

    @abstractmethod
    async def count(self, tenant_id: str) -> int:
        """Count total active chunks for a tenant."""


class InMemoryVectorStore(VectorStorePort):
    """
    Multi-tenant in-memory vector store with cosine similarity calculation.
    Suitable for development, testing, and isolated unit test executions.
    """

    def __init__(self):
        # tenant_id -> chunk_id -> DocumentChunk
        self._store: dict[str, dict[str, DocumentChunk]] = {}

    def add(self, embedding: list[float], metadata: dict) -> None:
        """Backward-compatible synchronous stub."""

    def query(self, vector: list[float], top_k: int = 5) -> list[dict]:
        """Backward-compatible synchronous stub."""
        return []

    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        added = 0
        for chunk in chunks:
            if not chunk.tenant_id:
                raise TenantContextMissingError("Chunk must contain tenant_id for vector storage")
            if chunk.embedding is None:
                raise VectorStoreError(
                    "Cannot store chunk without dense embedding", store_type="in_memory"
                )

            tid = chunk.tenant_id
            if tid not in self._store:
                self._store[tid] = {}

            self._store[tid][chunk.chunk_id] = chunk
            added += 1
        return added

    async def query_dense(
        self, vector: list[float], filter: RetrievalFilter, top_k: int = 10
    ) -> list[tuple[DocumentChunk, float]]:
        if not filter.tenant_id:
            raise TenantContextMissingError("Dense vector query requires authoritative tenant_id")

        tid = filter.tenant_id
        if tid not in self._store or not self._store[tid]:
            return []

        scored: list[tuple[DocumentChunk, float]] = []
        for chunk in self._store[tid].values():
            if chunk.embedding is None:
                continue

            # Filtering
            if filter.course_code and chunk.course_code != filter.course_code:
                continue
            if filter.document_ids and chunk.document_id not in filter.document_ids:
                continue
            if filter.authority_tiers and chunk.provenance:
                if chunk.provenance.authority not in filter.authority_tiers:
                    continue

            sim = cosine_similarity(vector, chunk.embedding)
            # Map cosine [-1.0, 1.0] to [0.0, 1.0]
            normalized_score = max(0.0, (sim + 1.0) / 2.0)

            if normalized_score >= filter.min_score:
                scored.append((chunk, normalized_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def delete_document(self, tenant_id: str, document_id: str) -> int:
        if tenant_id not in self._store:
            return 0
        to_del = [
            cid for cid, chunk in self._store[tenant_id].items() if chunk.document_id == document_id
        ]
        for cid in to_del:
            del self._store[tenant_id][cid]
        return len(to_del)

    async def count(self, tenant_id: str) -> int:
        return len(self._store.get(tenant_id, {}))


# The durable pgvector store is app.services.rag.pg_index.PgChunkIndex (P3-4).


# Alias for backward compatibility
VectorStore = InMemoryVectorStore
