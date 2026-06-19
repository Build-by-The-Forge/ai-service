from app.core.tenant import get_school_id
from app.services.rag.retrieval import RetrievalService


class TenantAwareRetrievalService:

    def __init__(self, retriever: RetrievalService | None = None):
        self.retriever = retriever or RetrievalService()

    async def retrieve(self, query: str) -> list[dict]:
        school_id = get_school_id()

        # Delegate to the tenant-aware retrieval service, which already scopes search by school_id.
        return await self.retriever.retrieve(query)
