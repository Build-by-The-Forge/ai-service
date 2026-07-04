from typing import Any

from app.services.rag.retrieval import RetrievalService


class TenantAwareRetrievalService(RetrievalService):

    def __init__(self, retriever: RetrievalService | None = None):
        self.retriever = retriever or RetrievalService()

    async def retrieve(self, query: str) -> list[dict[str, Any]]:
        # Delegate to the tenant-aware retrieval service, which already scopes search by school_id.
        return await self.retriever.retrieve(query)
