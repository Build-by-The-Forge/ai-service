"""The durable hybrid-retrieval index on Postgres (P3-4): document_chunks.

One table serves both sides of hybrid retrieval: the pgvector embedding
(dense) and the generated full-text vector (sparse), so a chunk's dense and
sparse entries are written, replaced and deleted together.

- Writes go through replace_asset only: an asset's previous chunks are
  deleted and the new ones inserted in one transaction, so re-indexing
  replaces, an identical re-run is idempotent, and a failure leaves the
  previous index untouched (never a partial one).
- Every read and delete filters on the verified tenant inside the SQL;
  nothing is read across tenants and filtered afterwards.
- Dense queries use the HNSW index with pgvector 0.8's iterative scan, so a
  tenant filter cannot starve the result set; sparse queries OR the query's
  terms (plainto_tsquery would AND them and miss most questions) and rank
  with ts_rank_cd, normalized to [0, 1).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from sqlalchemy import Float, delete, func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rag.contracts import ChunkProvenance, DocumentChunk, RetrievalFilter
from app.core.rag.errors import TenantContextMissingError, VectorStoreError
from app.db.models.document_chunk import DocumentChunk as ChunkRow
from app.services.rag.lexical import LexicalIndexPort
from app.services.rag.vector_store import VectorStorePort

if TYPE_CHECKING:
    from app.services.ingestion.chunking import Chunk

STORE = "pgvector"
_COLUMNS = (
    ChunkRow.chunk_id,
    ChunkRow.tenant_id,
    ChunkRow.asset_id,
    ChunkRow.run_number,
    ChunkRow.chunk_index,
    ChunkRow.page_start,
    ChunkRow.page_end,
    ChunkRow.content,
)
_SPARSE = text(
    "SELECT chunk_id, tenant_id, asset_id, run_number, chunk_index, page_start, page_end,"
    " content, ts_rank_cd(content_tsv, q.query, 32) AS score"
    " FROM document_chunks,"
    " (SELECT replace(plainto_tsquery('english', :query)::text, '&', '|')::tsquery AS query) q"
    " WHERE tenant_id = :tenant_id AND content_tsv @@ q.query"
    " AND (CAST(:asset_ids AS text[]) IS NULL OR asset_id = ANY(CAST(:asset_ids AS text[])))"
    " ORDER BY score DESC, chunk_id LIMIT :top_k"
)


def _tenant(tenant_id: str | None) -> str:
    if not tenant_id or not tenant_id.strip():
        raise TenantContextMissingError("the chunk index requires the verified tenant_id")
    return tenant_id


def _to_contract(row: Any) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=row.chunk_id,
        tenant_id=row.tenant_id,
        document_id=row.asset_id,
        document_title=row.asset_id,
        content=row.content,
        metadata={
            "run_number": row.run_number,
            "chunk_index": row.chunk_index,
            "page_start": row.page_start,
            "page_end": row.page_end,
        },
        provenance=ChunkProvenance(source_uri=f"asset:{row.asset_id}", page_number=row.page_start),
    )


class PgChunkIndex(VectorStorePort, LexicalIndexPort):
    store_type = STORE

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession] | None = None,
        *,
        ef_search: int = 100,
    ) -> None:
        self._session_factory = session_factory
        self._ef_search = int(ef_search)

    def _session(self) -> AsyncSession:
        if self._session_factory is None:
            from app.db.session import get_async_session_local

            self._session_factory = get_async_session_local()
        return self._session_factory()

    async def replace_asset(
        self,
        *,
        tenant_id: str,
        asset_id: str,
        run_number: int,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        embedding_model: str,
    ) -> int:
        tenant_id = _tenant(tenant_id)
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        rows = [
            ChunkRow(
                chunk_id=chunk.chunk_id,
                tenant_id=tenant_id,
                asset_id=asset_id,
                run_number=run_number,
                chunk_index=chunk.index,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                content=chunk.text,
                embedding=list(vector),
                embedding_model=embedding_model,
            )
            for chunk, vector in zip(chunks, embeddings)
        ]
        try:
            async with self._session() as db, db.begin():
                await db.execute(
                    delete(ChunkRow).where(
                        ChunkRow.tenant_id == tenant_id, ChunkRow.asset_id == asset_id
                    )
                )
                db.add_all(rows)
        except SQLAlchemyError as exc:
            raise VectorStoreError(f"{type(exc).__name__}: {exc}", store_type=STORE) from exc
        return len(rows)

    async def query_dense(
        self, vector: list[float], filter: RetrievalFilter, top_k: int = 10
    ) -> list[tuple[DocumentChunk, float]]:
        tenant_id = _tenant(filter.tenant_id)
        distance = ChunkRow.embedding.op("<=>", return_type=Float)(vector)
        stmt = (
            select(*_COLUMNS, distance.label("distance"))
            .where(ChunkRow.tenant_id == tenant_id)
            .order_by(distance)
            .limit(top_k)
        )
        if filter.document_ids:
            stmt = stmt.where(ChunkRow.asset_id.in_(filter.document_ids))
        try:
            async with self._session() as db, db.begin():
                await db.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
                await db.execute(text(f"SET LOCAL hnsw.ef_search = {self._ef_search}"))
                rows = (await db.execute(stmt)).all()
        except SQLAlchemyError as exc:
            raise VectorStoreError(f"{type(exc).__name__}: {exc}", store_type=STORE) from exc
        # relaxed_order may return near-ties slightly out of order.
        scored = [(_to_contract(r), max(0.0, 1.0 - float(r.distance) / 2.0)) for r in rows]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [(c, s) for c, s in scored if s >= filter.min_score]

    async def query_sparse(
        self, query: str, filter: RetrievalFilter, top_k: int = 10
    ) -> list[tuple[DocumentChunk, float]]:
        tenant_id = _tenant(filter.tenant_id)
        if not query.strip():
            return []
        params = {
            "query": query,
            "tenant_id": tenant_id,
            "asset_ids": list(filter.document_ids) if filter.document_ids else None,
            "top_k": top_k,
        }
        try:
            async with self._session() as db:
                rows = (await db.execute(_SPARSE, params)).all()
        except SQLAlchemyError as exc:
            raise VectorStoreError(f"{type(exc).__name__}: {exc}", store_type=STORE) from exc
        return [(_to_contract(r), float(r.score)) for r in rows if r.score >= filter.min_score]

    async def delete_document(self, tenant_id: str, document_id: str) -> int:
        tenant_id = _tenant(tenant_id)
        try:
            async with self._session() as db, db.begin():
                result = await db.execute(
                    delete(ChunkRow).where(
                        ChunkRow.tenant_id == tenant_id, ChunkRow.asset_id == document_id
                    )
                )
        except SQLAlchemyError as exc:
            raise VectorStoreError(f"{type(exc).__name__}: {exc}", store_type=STORE) from exc
        return int(getattr(result, "rowcount", 0) or 0)

    async def count(self, tenant_id: str) -> int:
        tenant_id = _tenant(tenant_id)
        async with self._session() as db:
            stmt = select(func.count()).select_from(ChunkRow).where(ChunkRow.tenant_id == tenant_id)
            return int((await db.execute(stmt)).scalar_one())

    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        raise VectorStoreError(
            "the durable index is written only through replace_asset (whole asset, one "
            "transaction)",
            store_type=STORE,
        )

    async def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        return await self.add_chunks(chunks)
