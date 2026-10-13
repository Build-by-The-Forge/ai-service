from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.tenant import get_tenant_id
from app.db.models.learning_memory import LearningMemory
from app.services.memory.reward import LearningReward

# A topic seen for the first time starts here before its first observation.
INITIAL_MASTERY = 0.5


def _current_tenant_id() -> str:
    # The verified context's tenant (a backend tenant UUID); raises when the
    # request has no verified scope rather than reading another tenant's rows.
    return get_tenant_id()


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

    async def query(self, user_id: str, topics: list[str] | None = None) -> list[LearningMemory]:
        """The user's memory in the request tenant, optionally for some topics."""
        statement = select(LearningMemory).where(
            LearningMemory.tenant_id == _current_tenant_id(),
            LearningMemory.user_id == user_id,
        )
        if topics:
            statement = statement.where(LearningMemory.topic.in_(topics))
        result = await self.db.execute(statement.order_by(LearningMemory.topic))
        return list(result.scalars().all())

    async def record(
        self, user_id: str, observations: list[tuple[str, bool]]
    ) -> list[LearningMemory]:
        """Apply graded outcomes in order (M5 design §3): mastery moves by the
        LearningReward delta, clamped to [0, 1]; attempts counts observations.
        One atomic upsert per observation, so concurrent submissions for the
        same topic never lose an update or create a second row."""
        tenant_id = _current_tenant_id()
        reward = LearningReward()
        now = datetime.now(UTC).isoformat()
        for topic, correct in observations:
            delta = reward.calculate(correct)
            statement = insert(LearningMemory).values(
                tenant_id=tenant_id,
                user_id=user_id,
                topic=topic,
                mastery_score=min(1.0, max(0.0, INITIAL_MASTERY + delta)),
                attempts=1,
                last_interaction=now,
            )
            statement = statement.on_conflict_do_update(
                constraint="uq_learning_memory_tenant_user_topic",
                set_={
                    "mastery_score": func.least(
                        1.0, func.greatest(0.0, LearningMemory.mastery_score + delta)
                    ),
                    "attempts": LearningMemory.attempts + 1,
                    "last_interaction": now,
                },
            )
            await self.db.execute(statement)
        await self.db.commit()
        return await self.query(user_id, sorted({topic for topic, _ in observations}))
