import asyncio
import time
from typing import Any

from app.core.rag.contracts import (
    ChunkProvenance,
    DocumentChunk,
    RetrievalFilter,
    RetrievalPolicy,
    RetrievedChunk,
    RetrievedContext,
)
from app.core.rag.errors import TenantContextMissingError
from app.services.rag.chunker import Chunker
from app.services.rag.embedder import EmbeddingPort, MockEmbeddingAdapter
from app.services.rag.lexical import BM25Index, LexicalIndexPort
from app.services.rag.vector_store import InMemoryVectorStore, VectorStorePort


class HybridRetrievalEngine:
    """
    Production Multi-Tenant Hybrid Retrieval Engine.
    Combines Dense Vector Similarity + Sparse BM25 Keyword Search using Reciprocal Rank Fusion (RRF).
    Guarantees strict tenant isolation and preserves verifiable citation provenance.
    """

    def __init__(
        self,
        embedder: EmbeddingPort | None = None,
        vector_store: VectorStorePort | None = None,
        lexical_index: LexicalIndexPort | None = None,
        chunker: Chunker | None = None,
    ):
        self.embedder = embedder or MockEmbeddingAdapter()
        self.vector_store = vector_store or InMemoryVectorStore()
        self.lexical_index = lexical_index or BM25Index()
        self.chunker = chunker or Chunker()

    async def index_document(
        self,
        tenant_id: str,
        document_id: str,
        document_title: str,
        text: str,
        course_code: str | None = None,
        topic: str | None = None,
        metadata: dict[str, Any] | None = None,
        provenance: ChunkProvenance | None = None,
    ) -> int:
        """
        Ingest, chunk, embed, and index a document across dense and sparse stores.
        """
        if not tenant_id or not tenant_id.strip():
            raise TenantContextMissingError("Indexing requires an authoritative tenant_id")

        chunks = self.chunker.chunk_document(
            tenant_id=tenant_id,
            document_id=document_id,
            document_title=document_title,
            text=text,
            course_code=course_code,
            topic=topic,
            metadata=metadata,
            provenance=provenance,
        )
        if not chunks:
            return 0

        # Generate embeddings in batch
        texts = [c.content for c in chunks]
        embeddings = await self.embedder.embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # Store in vector and lexical indexes
        await asyncio.gather(
            self.vector_store.add_chunks(chunks),
            self.lexical_index.index_chunks(chunks),
        )
        return len(chunks)

    async def retrieve(
        self,
        query: str,
        filter: RetrievalFilter,
        policy: RetrievalPolicy | None = None,
    ) -> RetrievedContext:
        """
        Execute dense and sparse hybrid retrieval with Reciprocal Rank Fusion.
        """
        start_time = time.perf_counter()

        if not filter.tenant_id or not filter.tenant_id.strip():
            raise TenantContextMissingError(
                "Retrieval operation requires an authoritative tenant_id"
            )

        policy = policy or RetrievalPolicy()

        # Generate query embedding
        query_vec = await self.embedder.embed_text(query)

        # Execute dense and sparse queries concurrently
        dense_task = self.vector_store.query_dense(
            vector=query_vec,
            filter=filter,
            top_k=policy.top_k * 2,
        )
        sparse_task = self.lexical_index.query_sparse(
            query=query,
            filter=filter,
            top_k=policy.top_k * 2,
        )

        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        # Reciprocal Rank Fusion (RRF)
        # RRF_Score(d) = w_dense / (rrf_k + rank_dense) + w_sparse / (rrf_k + rank_sparse)
        chunk_map: dict[str, DocumentChunk] = {}
        dense_scores: dict[str, float] = {}
        sparse_scores: dict[str, float] = {}
        rrf_scores: dict[str, float] = {}

        for rank, (chunk, score) in enumerate(dense_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            dense_scores[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (
                policy.dense_weight / (policy.rrf_k + rank)
            )

        for rank, (chunk, score) in enumerate(sparse_results, start=1):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            sparse_scores[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (
                policy.sparse_weight / (policy.rrf_k + rank)
            )

        # Sort candidate chunks by fused RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        retrieved_chunks: list[RetrievedChunk] = []
        cumulative_bytes = 0
        current_rank = 1

        for cid in sorted_cids:
            chunk = chunk_map[cid]
            chunk_bytes = len(chunk.content.encode("utf-8"))

            if cumulative_bytes + chunk_bytes > policy.max_context_bytes and retrieved_chunks:
                break

            rc = RetrievedChunk(
                chunk=chunk,
                dense_score=dense_scores.get(cid, 0.0),
                sparse_score=sparse_scores.get(cid, 0.0),
                hybrid_score=rrf_scores[cid],
                rank=current_rank,
            )
            retrieved_chunks.append(rc)
            cumulative_bytes += chunk_bytes
            current_rank += 1

            if len(retrieved_chunks) >= policy.top_k:
                break

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievedContext(
            query=query,
            tenant_id=filter.tenant_id,
            chunks=retrieved_chunks,
            retrieval_method="hybrid_dense_sparse_rrf",
            latency_ms=round(latency_ms, 2),
            total_candidates=len(sorted_cids),
        )


# Global singleton instance for app-wide hybrid retrieval
_default_retrieval_engine: HybridRetrievalEngine | None = None


def get_retrieval_engine() -> HybridRetrievalEngine:
    global _default_retrieval_engine
    if _default_retrieval_engine is None:
        _default_retrieval_engine = HybridRetrievalEngine()
    return _default_retrieval_engine


def set_retrieval_engine(engine: HybridRetrievalEngine) -> None:
    global _default_retrieval_engine
    _default_retrieval_engine = engine


# Backward-compatible RetrievalService facade
class RetrievalService:
    def __init__(self, engine: HybridRetrievalEngine | None = None):
        self.engine = engine or get_retrieval_engine()

    async def retrieve(self, query: str) -> list[dict[str, Any]]:
        from app.core import tenant

        # Scoped to the verified tenant only; no scope means no retrieval.
        tenant_id = tenant.get_tenant_id()
        filter_obj = RetrievalFilter(tenant_id=tenant_id)
        ctx = await self.engine.retrieve(query, filter=filter_obj)
        # Every item carries `text` (the contract field chat and the backend
        # read); real citations expose their content as `snippet`.
        citations = [{"text": c.get("snippet", ""), **c} for c in ctx.to_citation_summary()]
        if not citations:
            return [{"text": f"Tenant {tenant_id} retrieval placeholder for '{query}'"}]
        return citations
