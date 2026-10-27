"""PgChunkIndex against real PostgreSQL with pgvector (P3-4 acceptance).

Tenant isolation at the query, idempotent and atomic re-indexing, the
index surviving a fresh engine, and both retrieval sides.
"""

import os
import uuid

import pytest

from app.core.rag.contracts import RetrievalFilter
from app.core.rag.errors import TenantContextMissingError, VectorStoreError
from app.db.session import get_async_session_local, initialize_database, reset_engine
from app.services.ingestion.chunking import Chunk, chunk_id
from app.services.rag.pg_index import PgChunkIndex

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"), reason="needs PostgreSQL"
)
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXTS = [
    "Coulomb's law: like charges repel and unlike charges attract.",
    "The power rule: the derivative of x^n is n x^(n-1).",
    "Millikan's oil drop experiment measured the charge of the electron.",
]


def unit(axis: int) -> list[float]:
    vec = [0.0] * 384
    vec[axis] = 1.0
    return vec


def chunks(tenant: str, asset: str, texts=TEXTS, run: int = 1) -> list[Chunk]:
    return [
        Chunk(chunk_id(tenant, asset, run, i), i, t, page_start=i + 1, page_end=i + 1)
        for i, t in enumerate(texts)
    ]


async def index(store, tenant, asset, texts=TEXTS, run=1, vectors=None):
    return await store.replace_asset(
        tenant_id=tenant,
        asset_id=asset,
        run_number=run,
        chunks=chunks(tenant, asset, texts, run),
        embeddings=vectors or [unit(i) for i in range(len(texts))],
        embedding_model=MODEL,
    )


@pytest.fixture
async def store():
    await initialize_database()
    yield PgChunkIndex(get_async_session_local())
    await reset_engine()


@pytest.fixture
def tenants():
    return str(uuid.uuid4()), str(uuid.uuid4())


async def test_dense_returns_the_nearest_chunk_with_its_page(store, tenants):
    a, _ = tenants
    assert await index(store, a, "asset-1") == 3
    (top, score), *_ = await store.query_dense(unit(2), RetrievalFilter(tenant_id=a), top_k=3)
    assert top.content.startswith("Millikan") and score == pytest.approx(1.0)
    assert top.metadata["page_start"] == 3 and top.provenance.page_number == 3
    assert top.document_id == "asset-1" and top.tenant_id == a


async def test_sparse_ors_the_terms_and_stems(store, tenants):
    a, _ = tenants
    await index(store, a, "asset-1")
    hits = await store.query_sparse(
        "why do repelling charges matter xyzzy", RetrievalFilter(tenant_id=a), top_k=3
    )
    assert [c.content[:8] for c, _ in hits][:1] == ["Coulomb'"]
    assert all(0 < score < 1 for _, score in hits)
    assert await store.query_sparse("the and of", RetrievalFilter(tenant_id=a)) == []


async def test_tenant_b_never_retrieves_tenant_a(store, tenants):
    a, b = tenants
    await index(store, a, "asset-1")
    await index(store, b, "asset-2", texts=["Photosynthesis converts light to energy."])
    dense = await store.query_dense(unit(2), RetrievalFilter(tenant_id=b), top_k=10)
    sparse = await store.query_sparse("electron charge", RetrievalFilter(tenant_id=b), top_k=10)
    assert {c.tenant_id for c, _ in dense} == {b} and sparse == []
    # The same asset id under another tenant is another tenant's asset.
    assert await store.delete_document(b, "asset-1") == 0
    assert await store.count(a) == 3


async def test_reindexing_the_same_run_is_idempotent(store, tenants):
    a, _ = tenants
    await index(store, a, "asset-1")
    first = {c.chunk_id for c, _ in await store.query_dense(unit(0), RetrievalFilter(a), 10)}
    await index(store, a, "asset-1")
    again = {c.chunk_id for c, _ in await store.query_dense(unit(0), RetrievalFilter(a), 10)}
    assert first == again and await store.count(a) == 3


async def test_reindexing_replaces_the_asset_whole(store, tenants):
    a, _ = tenants
    await index(store, a, "asset-1")
    await index(store, a, "asset-1", texts=["A single new chunk."], run=2)
    rows = await store.query_dense(unit(0), RetrievalFilter(a), 10)
    assert [c.content for c, _ in rows] == ["A single new chunk."]
    assert rows[0][0].metadata["run_number"] == 2


async def test_a_failed_reindex_leaves_the_previous_index_untouched(store, tenants):
    a, _ = tenants
    await index(store, a, "asset-1")
    with pytest.raises((VectorStoreError, ValueError)):
        await index(store, a, "asset-1", run=2, vectors=[unit(0), unit(1), [1.0, 0.0]])
    assert await store.count(a) == 3
    rows = await store.query_dense(unit(0), RetrievalFilter(a), 10)
    assert {c.metadata["run_number"] for c, _ in rows} == {1}


async def test_the_asset_filter_is_applied_in_the_query(store, tenants):
    a, _ = tenants
    await index(store, a, "asset-1")
    await index(store, a, "asset-2", texts=["Charge is conserved in every interaction."])
    only = RetrievalFilter(tenant_id=a, document_ids=["asset-2"])
    assert {c.document_id for c, _ in await store.query_dense(unit(0), only, 10)} == {"asset-2"}
    assert {c.document_id for c, _ in await store.query_sparse("charge", only, 10)} == {"asset-2"}


async def test_the_index_survives_a_fresh_engine(tenants):
    a, _ = tenants
    await initialize_database()
    await index(PgChunkIndex(get_async_session_local()), a, "asset-1")
    await reset_engine()
    await initialize_database()
    try:
        assert await PgChunkIndex(get_async_session_local()).count(a) == 3
    finally:
        await reset_engine()


async def test_every_operation_requires_a_tenant(store):
    with pytest.raises(TenantContextMissingError):
        await store.query_dense(unit(0), RetrievalFilter(tenant_id=" "))
    with pytest.raises(TenantContextMissingError):
        await store.query_sparse("x", RetrievalFilter(tenant_id=""))
    with pytest.raises(TenantContextMissingError):
        await store.count("")
    with pytest.raises(TenantContextMissingError):
        await index(store, "", "asset-1")


async def test_chunks_are_written_only_whole_asset(store):
    with pytest.raises(VectorStoreError, match="only through replace_asset"):
        await store.add_chunks([])
