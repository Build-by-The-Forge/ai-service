from app.core import tenant


class RetrievalService:
    async def retrieve(self, query: str) -> list[dict]:
        school_id = tenant.get_school_id()

        # Tenant-aware retrieval placeholder.
        # Replace with a real vector search that filters by school_id.
        return [{"text": f"Tenant {school_id} retrieval placeholder for '{query}'"}]
