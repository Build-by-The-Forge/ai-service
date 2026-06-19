from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models.learning_memory import LearningMemory


class MemoryService:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_memory(self, user_id: str):
        result = await self.db.execute(
            select(LearningMemory).where(LearningMemory.user_id == user_id)
        )

        return result.scalars().all()

    async def update_mastery(self, user_id: str, topic: str, delta: float):
        result = await self.db.execute(
            select(LearningMemory).where(
                LearningMemory.user_id == user_id,
                LearningMemory.topic == topic,
            )
        )

        record = result.scalar_one_or_none()

        if not record:
            record = LearningMemory(
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
