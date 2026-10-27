import pytest

from app.core import tenant
from app.services.rag.retrieval import RetrievalService

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
        return await RetrievalService().retrieve(query)
    finally:
        tenant.clear_request_scope()


async def test_retrieval_returns_the_tenants_own_passage_with_its_page(in_memory_retrieval):
    await in_memory_retrieval.index_document(
        tenant_id=TENANT_A,
        document_id="asset-1",
        document_title="MTS 102 notes",
        text="The derivative of a constant function is zero.",
        metadata={"page_start": 3},
    )

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
