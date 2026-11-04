"""The scope filter on the durable index, against real PostgreSQL (M3-D).

Only `institutional` documents are returned, on the dense and the keyword
side, and the filter runs inside the SQL before the limit. A document with
no scope row, which is every document indexed before asset_scopes existed,
is `unknown` and never returned.
"""

import os
import uuid

import pytest

from app.core.rag.contracts import RetrievalFilter
from app.db.session import get_async_session_local, initialize_database, reset_engine
from app.services.ingestion.chunking import Chunk, chunk_id
from app.services.rag.pg_index import PgChunkIndex
from tests.rag_support import stored_scope

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"), reason="needs PostgreSQL"
)
TEXT = "Kirchhoff's current law: the charge entering a junction equals the charge leaving it."


def unit(axis: int) -> list[float]:
    vec = [0.0] * 384
    vec[axis] = 1.0
    return vec


@pytest.fixture
async def store():
    await initialize_database()
    yield PgChunkIndex(get_async_session_local())
    await reset_engine()


async def index(store, tenant, asset, *, run=1, text=TEXT, vector=None) -> None:
    await store.raise_fence(tenant_id=tenant, asset_id=asset, run_number=run, attempt=1)
    await store.replace_asset(
        tenant_id=tenant,
        asset_id=asset,
        run_number=run,
        attempt=1,
        chunks=[Chunk(chunk_id(tenant, asset, run, 0), 0, text, 1, 1)],
        embeddings=[vector or unit(0)],
        embedding_model="mock",
    )


async def found(store, tenant) -> set[str]:
    """The documents each side of retrieval returns for the tenant."""
    filter = RetrievalFilter(tenant_id=tenant, denied_assets=frozenset())
    dense = await store.query_dense(unit(0), filter, top_k=50)
    sparse = await store.query_sparse("charge junction", filter, top_k=50)
    return {c.document_id for c, _ in dense} | {c.document_id for c, _ in sparse}


def asset() -> str:
    return f"asset-{uuid.uuid4().hex[:12]}"


async def test_only_institutional_documents_are_returned(store):
    tenant = str(uuid.uuid4())
    shared, personal, unknown, legacy = asset(), asset(), asset(), asset()
    for document in (shared, personal, unknown, legacy):
        await index(store, tenant, document)
    await stored_scope(tenant, shared, "institutional")
    await stored_scope(tenant, personal, "personal")
    await stored_scope(tenant, unknown, "unknown")
    # `legacy` has no row, like every document indexed before the migration.

    for _ in range(7):  # past five executions, where a generic plan could differ
        assert await found(store, tenant) == {shared}


async def test_a_document_without_a_scope_row_is_never_returned(store):
    tenant = str(uuid.uuid4())
    document = asset()
    await index(store, tenant, document)
    assert await found(store, tenant) == set()
    assert await store.count(tenant) == 1  # indexed, just not retrievable


async def test_lowering_a_document_removes_it_from_the_next_query(store):
    tenant = str(uuid.uuid4())
    document = asset()
    await index(store, tenant, document)
    await stored_scope(tenant, document, "institutional", version=1)
    assert await found(store, tenant) == {document}
    await stored_scope(tenant, document, "unknown", version=2)
    assert await found(store, tenant) == set()


async def test_reprocessed_chunks_take_the_documents_current_scope(store):
    tenant = str(uuid.uuid4())
    document = asset()
    await index(store, tenant, document, run=1)
    await stored_scope(tenant, document, "personal", version=1)
    # A later run, or chunks a job in flight writes late, carry no scope of
    # their own: the document's row decides.
    await index(store, tenant, document, run=2)
    assert await found(store, tenant) == set()
    await stored_scope(tenant, document, "institutional", version=2)
    assert await found(store, tenant) == {document}
    await index(store, tenant, document, run=3)
    assert await found(store, tenant) == {document}


async def test_the_scope_is_the_documents_own_tenants(store):
    owner, other = str(uuid.uuid4()), str(uuid.uuid4())
    document = asset()
    await index(store, owner, document)
    # A row for the same asset id under another tenant scopes nothing of the owner's.
    await stored_scope(other, document, "institutional")
    assert await found(store, owner) == set()
    assert await found(store, other) == set()


async def test_the_filter_runs_before_the_limit(store):
    # Many excluded chunks nearer the query than the one eligible chunk:
    # filtering after the limit would return nothing.
    tenant = str(uuid.uuid4())
    eligible = asset()
    await index(store, tenant, eligible, vector=unit(5))
    await stored_scope(tenant, eligible, "institutional")
    for i in range(40):
        near = [0.0] * 384
        near[0], near[5] = 1.0, 0.01 * (i + 1)
        await index(store, tenant, asset(), text=f"charge junction {i}", vector=near)

    filter = RetrievalFilter(tenant_id=tenant, denied_assets=frozenset())
    dense = await store.query_dense(unit(0), filter, top_k=1)
    sparse = await store.query_sparse("charge junction", filter, top_k=1)
    assert [c.document_id for c, _ in dense] == [eligible]
    assert [c.document_id for c, _ in sparse] == [eligible]


async def test_a_denied_document_is_excluded_whatever_its_scope(store):
    # Lowered by the backend, not yet acknowledged: still institutional here.
    tenant = str(uuid.uuid4())
    kept, denied = asset(), asset()
    for document in (kept, denied):
        await index(store, tenant, document)
        await stored_scope(tenant, document, "institutional")
    filter = RetrievalFilter(tenant_id=tenant, denied_assets=frozenset({denied}))
    for _ in range(7):  # past five executions
        dense = await store.query_dense(unit(0), filter, top_k=1)
        sparse = await store.query_sparse("charge junction", filter, top_k=1)
        # Excluded before the limit: top_k=1 still finds the other document.
        assert [c.document_id for c, _ in dense] == [kept]
        assert [c.document_id for c, _ in sparse] == [kept]
    assert await found(store, tenant) == {kept, denied}  # nothing changed in the index
