import pytest

from app.core import tenant
from app.services.rag.retrieval import RetrievalService
from tests.rag_support import institutional


@pytest.fixture(autouse=True)
def no_scope():
    tenant.clear_request_scope()
    yield
    tenant.clear_request_scope()


@pytest.mark.asyncio
async def test_retrieval_service_uses_the_verified_tenant(in_memory_retrieval, in_memory_scopes):
    await in_memory_retrieval.index_document(
        tenant_id="11111111-1111-4111-8111-111111111111",
        document_id="asset-1",
        document_title="Greetings",
        text="hello and welcome to the course",
    )
    await institutional(in_memory_scopes, "11111111-1111-4111-8111-111111111111", "asset-1")
    tenant.set_request_scope("11111111-1111-4111-8111-111111111111", "user-1")

    result = (await RetrievalService().retrieve("hello", frozenset())).items

    assert result[0]["text"] == "hello and welcome to the course"


@pytest.mark.asyncio
async def test_retrieval_without_a_verified_tenant_fails_closed():
    with pytest.raises(tenant.RequestScopeMissingError):
        await RetrievalService().retrieve("hello", frozenset())
