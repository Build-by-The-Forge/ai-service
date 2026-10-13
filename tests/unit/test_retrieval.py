import pytest

from app.core import tenant
from app.services.rag.retrieval import RetrievalService


@pytest.fixture(autouse=True)
def no_scope():
    tenant.clear_request_scope()
    yield
    tenant.clear_request_scope()


@pytest.mark.asyncio
async def test_retrieval_service_uses_the_verified_tenant():
    tenant.set_request_scope("11111111-1111-4111-8111-111111111111", "user-1")

    result = await RetrievalService().retrieve("hello")

    assert result[0]["text"].startswith("Tenant 11111111-1111-4111-8111-111111111111")


@pytest.mark.asyncio
async def test_retrieval_without_a_verified_tenant_fails_closed():
    with pytest.raises(tenant.RequestScopeMissingError):
        await RetrievalService().retrieve("hello")
