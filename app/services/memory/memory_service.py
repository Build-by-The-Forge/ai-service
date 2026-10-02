from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.tenant import get_school_id
from app.db.models.learning_memory import LearningMemory


def _current_tenant_id() -> str:
    # Temporary bridge: the request tenant context is still the legacy integer
    # school id. Phase 1 Item 3 replaces it with the authenticated backend
    # tenant id; until then it is stored in tenant_id as a string.
    return str(get_school_id())


class MemoryService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_memory(self, user_id: str):
        tenant_id = _current_tenant_id()
        result = await self.db.execute(
            select(LearningMemory).where(
                LearningMemory.user_id == user_id,
                LearningMemory.tenant_id == tenant_id,
            )
        )

        return result.scalars().all()

    async def update_mastery(self, user_id: str, topic: str, delta: float):
        tenant_id = _current_tenant_id()
        result = await self.db.execute(
            select(LearningMemory).where(
                LearningMemory.user_id == user_id,
                LearningMemory.topic == topic,
                LearningMemory.tenant_id == tenant_id,
            )
        )

        record = result.scalar_one_or_none()

        if not record:
            record = LearningMemory(
                tenant_id=tenant_id,
                user_id=user_id,
                topic=topic,
                mastery_score=0.0,
                attempts=0,
                last_interaction="",
            )
            self.db.add(record)

        record.mastery_score += delta
        record.attempts += 1

        await self.db.commit()

        return record
