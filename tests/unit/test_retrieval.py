import pytest

from app.services.rag.retrieval import RetrievalService
from app.core import tenant


@pytest.mark.asyncio
async def test_retrieval_service_uses_current_school_id(monkeypatch):
    monkeypatch.setattr(tenant, "get_school_id", lambda: 42)

    service = RetrievalService()
    result = await service.retrieve("hello")

    assert result[0]["text"].startswith("Tenant 42")
