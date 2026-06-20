import pytest

from app.services.rag.retrieval import RetrievalService
import app.core.tenant as tenant


@pytest.mark.asyncio
async def test_rag_retrieval_monkeypatched_tenant(monkeypatch):
    # Monkeypatch tenant to return a school id for tenant-aware retrieval
    monkeypatch.setattr(tenant, "get_school_id", lambda: 1)

    retriever = RetrievalService()
    res = await retriever.retrieve("sample query")

    assert isinstance(res, list)
