import pytest

from app.core import tenant
from app.services.rag.retrieval import RetrievalService


@pytest.mark.asyncio
async def test_retrieval_service_uses_current_tenant():
    tenant.set_request_scope("11111111-1111-4111-8111-111111111111")
    try:
        result = await RetrievalService().retrieve("algebra")
    finally:
        tenant.clear_request_scope()

    assert (
        result[0]["text"]
        == "Tenant 11111111-1111-4111-8111-111111111111 retrieval placeholder for 'algebra'"
    )
