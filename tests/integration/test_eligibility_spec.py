"""M4 as a specification: the AI service's side of retrieval eligibility
(build-backend docs/m4-retrieval-eligibility-design.md, sections 3.2 to 3.5).

Every test is a strict xfail on real PostgreSQL: it fails today for the reason
in its marker, and when M4-B makes it pass the strict marker turns that into a
failure, so the marker is removed in the same commit. No existing test is
changed.

The seam is one new, required, keyword-only field on ``RetrievalFilter``:
``allowed``, a frozenset of ``(asset_id, run_number, attempt)``. The same
field is required on every retrieving request body (``RetrievingRequest``).
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
M4B = "M4-B phase 2 (design section 7)"
TEXT = "Kirchhoff's current law: the charge entering a junction equals the charge leaving it."


def strict(reason: str):
    return pytest.mark.xfail(strict=True, reason=f"{reason}: {M4B}")


def unit(axis: int) -> list[float]:
    vec = [0.0] * 384
    vec[axis] = 1.0
    return vec


@pytest.fixture
async def store():
    await initialize_database()
    yield PgChunkIndex(get_async_session_local())
    await reset_engine()


def asset() -> str:
    return f"asset-{uuid.uuid4().hex[:12]}"


async def index(store, tenant, asset_id, *, run=1, attempt=1) -> None:
    await store.raise_fence(tenant_id=tenant, asset_id=asset_id, run_number=run, attempt=attempt)
    await store.replace_asset(
        tenant_id=tenant,
        asset_id=asset_id,
        run_number=run,
        attempt=attempt,
        chunks=[Chunk(chunk_id(tenant, asset_id, run, 0), 0, TEXT, 1, 1)],
        embeddings=[unit(0)],
        embedding_model="mock",
    )


async def found(store, tenant, allowed) -> set[tuple[str, int, int]]:
    """What each side of retrieval returns for the tenant, as exact versions."""
    filter = RetrievalFilter(
        tenant_id=tenant, denied_assets=frozenset(), allowed=frozenset(allowed)
    )
    dense = await store.query_dense(unit(0), filter, top_k=50)
    sparse = await store.query_sparse("charge junction", filter, top_k=50)
    return {
        (c.document_id, c.metadata["run_number"], c.metadata["attempt"]) for c, _ in dense + sparse
    }


@strict("RetrievalFilter has no allow-list; absence still means unfiltered")
def test_an_allow_list_is_required_never_defaulted():
    with pytest.raises(TypeError):
        RetrievalFilter(tenant_id="t", denied_assets=frozenset())  # type: ignore[call-arg]


@strict("an empty list is not a filter today")
async def test_an_empty_allow_list_returns_nothing_even_for_an_institutional_document(store):
    tenant, doc = str(uuid.uuid4()), asset()
    await index(store, tenant, doc)
    await stored_scope(tenant, doc, "institutional")
    assert await found(store, tenant, set()) == set()


@strict("only the document's scope gates retrieval today")
async def test_only_the_listed_document_is_returned(store):
    tenant, listed, unlisted = str(uuid.uuid4()), asset(), asset()
    for doc in (listed, unlisted):
        await index(store, tenant, doc)
        await stored_scope(tenant, doc, "institutional")
    assert await found(store, tenant, {(listed, 1, 1)}) == {(listed, 1, 1)}


@strict("the run and attempt are not part of the match today")
@pytest.mark.parametrize("run,attempt", [(2, 1), (1, 2)])
async def test_a_listing_for_another_run_or_attempt_matches_no_chunk(store, run, attempt):
    tenant, doc = str(uuid.uuid4()), asset()
    await index(store, tenant, doc, run=1, attempt=1)
    await stored_scope(tenant, doc, "institutional")
    assert await found(store, tenant, {(doc, run, attempt)}) == set()


@strict("a personal document is never retrievable today, even listed for its owner")
async def test_a_personal_document_is_returned_only_when_listed(store):
    tenant, doc = str(uuid.uuid4()), asset()
    await index(store, tenant, doc)
    await stored_scope(tenant, doc, "personal")
    assert await found(store, tenant, set()) == set()
    assert await found(store, tenant, {(doc, 1, 1)}) == {(doc, 1, 1)}


@strict("the allow-list does not exist to be intersected with the scope")
@pytest.mark.parametrize("scope", ["unknown", None])
async def test_a_listed_document_without_a_retrievable_scope_is_never_returned(store, scope):
    tenant, doc = str(uuid.uuid4()), asset()
    await index(store, tenant, doc)
    if scope:
        await stored_scope(tenant, doc, scope)
    assert await found(store, tenant, {(doc, 1, 1)}) == set()


@strict("the allow-list does not exist")
async def test_a_listed_asset_of_another_tenant_matches_nothing(store):
    tenant_a, tenant_b, doc = str(uuid.uuid4()), str(uuid.uuid4()), asset()
    await index(store, tenant_a, doc)
    await stored_scope(tenant_a, doc, "institutional")
    assert await found(store, tenant_b, {(doc, 1, 1)}) == set()
    assert await found(store, tenant_a, {(doc, 1, 1)}) == {(doc, 1, 1)}  # and it is A's alone


@strict("a document that is not listed is excluded from the dense query only by its scope today")
async def test_a_denied_document_stays_out_even_when_listed(store):
    tenant, doc = str(uuid.uuid4()), asset()
    await index(store, tenant, doc)
    await stored_scope(tenant, doc, "institutional")
    filter = RetrievalFilter(
        tenant_id=tenant, denied_assets=frozenset({doc}), allowed=frozenset({(doc, 1, 1)})
    )
    assert await store.query_dense(unit(0), filter, top_k=50) == []
    assert await store.query_sparse("charge junction", filter, top_k=50) == []


@strict("dispatch raises a fence for an asset another tenant holds (design 2.4, hardening)")
async def test_a_dispatch_for_an_asset_another_tenant_holds_is_refused(store):
    from app.services.rag.scopes import AssetHeldByAnotherTenantError

    tenant_a, tenant_b, doc = str(uuid.uuid4()), str(uuid.uuid4()), asset()
    await store.raise_fence(tenant_id=tenant_a, asset_id=doc, run_number=1, attempt=1)
    with pytest.raises(AssetHeldByAnotherTenantError):
        await store.raise_fence(tenant_id=tenant_b, asset_id=doc, run_number=1, attempt=1)


@strict("the retrieving request bodies carry no allow-list")
def test_every_retrieving_request_requires_a_bounded_allow_list():
    from app.schemas.ai import (
        ChatRequest,
        ExplainRequest,
        GenerateQuestionsRequest,
        RetrieveRequest,
    )

    for model in (ChatRequest, ExplainRequest, GenerateQuestionsRequest, RetrieveRequest):
        schema = model.model_json_schema()
        assert "allowed" in schema["required"], model.__name__
        assert schema["properties"]["allowed"]["maxItems"] == 5000, model.__name__
