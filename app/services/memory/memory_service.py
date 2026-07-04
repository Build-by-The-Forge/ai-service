from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models.learning_memory import LearningMemory
from app.core.tenant import get_school_id


class MemoryService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_memory(self, user_id: str):
        school_id = get_school_id()
        result = await self.db.execute(
            select(LearningMemory).where(
                LearningMemory.user_id == user_id,
                LearningMemory.school_id == school_id,
            )
        )

        return result.scalars().all()

    async def update_mastery(self, user_id: str, topic: str, delta: float):
        school_id = get_school_id()
        result = await self.db.execute(
            select(LearningMemory).where(
                LearningMemory.user_id == user_id,
                LearningMemory.topic == topic,
                LearningMemory.school_id == school_id,
            )
        )

        record = result.scalar_one_or_none()

        if not record:
            record = LearningMemory(
                school_id=school_id,
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
