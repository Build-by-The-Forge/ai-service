import pytest

from app.core import tenant
from app.services.rag.retrieval import RetrievalService


@pytest.mark.asyncio
async def test_retrieval_service_uses_current_tenant_school_id(monkeypatch):
    tenant.set_school_id(123)
    service = RetrievalService()

    result = await service.retrieve("algebra")

    assert result[0]["text"] == "Tenant 123 retrieval placeholder for 'algebra'"
