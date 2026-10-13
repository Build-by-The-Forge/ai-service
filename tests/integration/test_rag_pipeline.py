import pytest

from app.core import tenant
from app.services.rag.retrieval import RetrievalService


@pytest.mark.asyncio
async def test_rag_retrieval_monkeypatched_tenant(monkeypatch):
    # Tenant-aware retrieval reads the verified tenant id
    monkeypatch.setattr(tenant, "get_tenant_id", lambda: "tenant-1")

    retriever = RetrievalService()
    res = await retriever.retrieve("sample query")

    assert isinstance(res, list)
