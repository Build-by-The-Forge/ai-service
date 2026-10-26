"""document_chunks against real PostgreSQL with pgvector (P3-4).

Needs DATABASE_URL pointing at PostgreSQL with the vector extension
available (the pinned pgvector image in CI and compose).
"""

import os
import uuid

import pytest
from sqlalchemy import select, text

from app.db.models.document_chunk import EMBEDDING_DIMENSION, DocumentChunk
from app.db.session import get_async_session_local, initialize_database, reset_engine

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"), reason="needs PostgreSQL"
)


@pytest.fixture
async def tenant():
    await initialize_database()
    yield str(uuid.uuid4())
    await reset_engine()


def unit(axis: int) -> list[float]:
    vec = [0.0] * EMBEDDING_DIMENSION
    vec[axis] = 1.0
    return vec


def row(tenant_id: str, index: int, content: str, embedding: list[float]) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        asset_id="asset-1",
        run_number=1,
        chunk_index=index,
        page_start=index + 1,
        page_end=index + 1,
        content=content,
        embedding=embedding,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )


async def test_embedding_round_trips_and_ranks_by_cosine(tenant):
    async with get_async_session_local()() as db:
        db.add_all(
            [
                row(tenant, 0, "Coulomb's law", unit(0)),
                row(tenant, 1, "the power rule", unit(1)),
            ]
        )
        await db.commit()

        stored = (
            await db.execute(select(DocumentChunk).where(DocumentChunk.tenant_id == tenant))
        ).scalars()
        assert sorted(c.embedding.index(1.0) for c in stored) == [0, 1]

        nearest = (
            await db.execute(
                select(DocumentChunk.content)
                .where(DocumentChunk.tenant_id == tenant)
                .order_by(DocumentChunk.embedding.op("<=>")(unit(1)))
                .limit(1)
            )
        ).scalar_one()
        assert nearest == "the power rule"


async def test_full_text_vector_is_generated_and_stemmed(tenant):
    async with get_async_session_local()() as db:
        db.add(row(tenant, 0, "Charges repel when they are alike.", unit(2)))
        await db.commit()
        match = (
            await db.execute(
                select(DocumentChunk.chunk_index).where(
                    DocumentChunk.tenant_id == tenant,
                    DocumentChunk.content_tsv.op("@@")(
                        text("plainto_tsquery('english', 'repelling charge')")
                    ),
                )
            )
        ).scalar_one()
        assert match == 0


async def test_wrong_dimension_is_refused_before_the_database(tenant):
    async with get_async_session_local()() as db:
        db.add(row(tenant, 0, "short vector", [1.0, 0.0]))
        with pytest.raises(Exception, match="expected 384 dimensions"):
            await db.commit()
        await db.rollback()


async def test_indexes_are_tenant_first(tenant):
    async with get_async_session_local()() as db:
        defs = dict(
            (
                await db.execute(
                    text(
                        "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'document_chunks'"
                    )
                )
            ).all()
        )
    assert "(tenant_id, asset_id)" in defs["ix_document_chunks_tenant_asset"]
    assert "USING gin (content_tsv)" in defs["ix_document_chunks_tsv"]
    assert "USING hnsw (embedding vector_cosine_ops)" in defs["ix_document_chunks_embedding"]
