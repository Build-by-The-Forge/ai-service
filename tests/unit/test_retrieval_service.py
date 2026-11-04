import pytest

from app.core import tenant
from app.core.rag.contracts import RetrievalFilter
from app.core.rag.scope import KnowledgeScope
from app.services.rag.embedder import MockEmbeddingAdapter
from app.services.rag.lexical import BM25Index
from app.services.rag.retrieval import HybridRetrievalEngine, RetrievalService
from app.services.rag.vector_store import InMemoryVectorStore
from tests.rag_support import institutional

TENANT_A = "11111111-1111-4111-8111-111111111111"
TENANT_B = "22222222-2222-4222-8222-222222222222"


@pytest.fixture(autouse=True)
def no_scope():
    tenant.clear_request_scope()
    yield
    tenant.clear_request_scope()


async def retrieve_as(tenant_id: str, query: str) -> list[dict]:
    tenant.set_request_scope(tenant_id)
    try:
        return await RetrievalService().retrieve(query, frozenset())
    finally:
        tenant.clear_request_scope()


async def test_retrieval_returns_the_tenants_own_passage_with_its_page(
    in_memory_retrieval, in_memory_scopes
):
    await in_memory_retrieval.index_document(
        tenant_id=TENANT_A,
        document_id="asset-1",
        document_title="MTS 102 notes",
        text="The derivative of a constant function is zero.",
        metadata={"page_start": 3},
    )
    await institutional(in_memory_scopes, TENANT_A, "asset-1")

    [hit] = await retrieve_as(TENANT_A, "derivative of a constant")

    assert hit["text"].startswith("The derivative of a constant")
    assert hit["document_title"] == "MTS 102 notes"


async def test_another_tenant_retrieves_nothing_and_nothing_is_invented(in_memory_retrieval):
    await in_memory_retrieval.index_document(
        tenant_id=TENANT_A,
        document_id="asset-1",
        document_title="MTS 102 notes",
        text="The derivative of a constant function is zero.",
    )

    # No stand-in passage: a tenant with no evidence gets an empty result.
    assert await retrieve_as(TENANT_B, "derivative of a constant") == []


@pytest.mark.parametrize("scope", [None, "personal", "unknown"])
async def test_a_document_that_is_not_institutional_is_never_retrieved(
    in_memory_retrieval, in_memory_scopes, scope
):
    await in_memory_retrieval.index_document(
        tenant_id=TENANT_A,
        document_id="asset-private",
        document_title="A student's notes",
        text="The derivative of a constant function is zero.",
    )
    if scope is not None:
        await in_memory_scopes.assign(
            tenant_id=TENANT_A,
            asset_id="asset-private",
            scope=KnowledgeScope(scope),
            version=1,
        )

    assert await retrieve_as(TENANT_A, "derivative of a constant") == []
    tenant.set_request_scope(TENANT_A)
    try:
        assert (
            await RetrievalService().evidence("derivative of a constant", frozenset())
        ).passages == ()
    finally:
        tenant.clear_request_scope()


async def test_stores_without_scopes_return_nothing():
    # No scope store: no document has a scope, so nothing is returned.
    bare = HybridRetrievalEngine(MockEmbeddingAdapter(), InMemoryVectorStore(), BM25Index())
    await bare.index_document(
        tenant_id=TENANT_A, document_id="asset-1", document_title="t", text="hello world"
    )
    assert (
        await bare.retrieve(
            "hello", filter=RetrievalFilter(tenant_id=TENANT_A, denied_assets=frozenset())
        )
    ).chunks == []
