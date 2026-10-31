from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.learning_memory_gate import LearningMemoryDisabled
from app.core.metrics import learning_memory_refusals_total
from app.core.tenant import get_tenant_id
from app.db.models.learning_memory import (
    LearningMemory,
    LearningMemoryContribution,
    LearningMemoryEvent,
)
from app.services.memory.reward import LearningReward

# A topic seen for the first time starts here before its first observation.
INITIAL_MASTERY = 0.5


def _current_tenant_id() -> str:
    # The verified context's tenant (a backend tenant UUID); raises when the
    # request has no verified scope rather than reading another tenant's rows.
    return get_tenant_id()


class LearningMemoryEventConflict(Exception):
    """An event ID came back with another student or a different body."""


class LearningMemoryHistoryIncomplete(Exception):
    """A topic's row predates contributions, so it can't be rebuilt from them."""


def fold_mastery(outcomes: list[bool]) -> float:
    """The aggregate's value for these outcomes, in order. Clamping depends on
    order, so a recompute must apply them in the order they were recorded."""
    reward = LearningReward()
    mastery = INITIAL_MASTERY
    for correct in outcomes:
        mastery = min(1.0, max(0.0, mastery + reward.calculate(correct)))
    return mastery


def _require_gate_open(operation: str) -> None:
    # Before the tenant lookup and any query: every read and write of learning
    # memory goes through this class, so this is the one enforcement point.
    if not settings.learning_memory_enabled:
        learning_memory_refusals_total.labels(operation=operation).inc()
        raise LearningMemoryDisabled


class MemoryService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_user_memory(self, user_id: str):
        _require_gate_open("read")
        tenant_id = _current_tenant_id()
        result = await self.db.execute(
            select(LearningMemory).where(
                LearningMemory.user_id == user_id,
                LearningMemory.tenant_id == tenant_id,
            )
        )

        return result.scalars().all()

    async def query(self, user_id: str, topics: list[str] | None = None) -> list[LearningMemory]:
        """The user's memory in the request tenant, optionally for some topics."""
        _require_gate_open("read")
        statement = select(LearningMemory).where(
            LearningMemory.tenant_id == _current_tenant_id(),
            LearningMemory.user_id == user_id,
        )
        if topics:
            statement = statement.where(LearningMemory.topic.in_(topics))
        result = await self.db.execute(statement.order_by(LearningMemory.topic))
        return list(result.scalars().all())

    async def record(
        self,
        user_id: str,
        observations: list[tuple[str, bool]],
        *,
        event_id: str | None = None,
        source: tuple[str, str] | None = None,
    ) -> list[LearningMemory]:
        """Apply graded outcomes in order (M5 design §3): mastery moves by the
        LearningReward delta, clamped to [0, 1]; attempts counts observations.
        The event and its effect commit together, so a repeated event_id
        changes nothing and returns success. With no event_id nothing can
        recognise a repeat, so each call counts: at-least-once with duplicates."""
        _require_gate_open("write")
        tenant_id = _current_tenant_id()
        await self._lock_student(tenant_id, user_id)
        now = datetime.now(UTC)
        topics = sorted({topic for topic, _ in observations})
        event_row_id = await self._insert_event(tenant_id, user_id, event_id, source, now)
        if event_row_id is None:
            await self._check_repeat(tenant_id, user_id, event_id or "", observations)
            await self.db.commit()
            return await self.query(user_id, topics)

        await self._add_contributions(tenant_id, user_id, event_row_id, observations)
        reward = LearningReward()
        stamp = now.isoformat()
        for topic, correct in observations:
            delta = reward.calculate(correct)
            statement = insert(LearningMemory).values(
                tenant_id=tenant_id,
                user_id=user_id,
                topic=topic,
                mastery_score=min(1.0, max(0.0, INITIAL_MASTERY + delta)),
                attempts=1,
                last_interaction=stamp,
                # Not in the update below: a legacy row stays incomplete.
                history_complete=True,
            )
            statement = statement.on_conflict_do_update(
                constraint="uq_learning_memory_tenant_user_topic",
                set_={
                    "mastery_score": func.least(
                        1.0, func.greatest(0.0, LearningMemory.mastery_score + delta)
                    ),
                    "attempts": LearningMemory.attempts + 1,
                    "last_interaction": stamp,
                },
            )
            await self.db.execute(statement)
        await self.db.commit()
        return await self.query(user_id, topics)

    async def revise_event(
        self, user_id: str, event_id: str, observations: list[tuple[str, bool]]
    ) -> list[LearningMemory]:
        """Replace one event's contributions: no duplicate, and the others stay."""
        _require_gate_open("write")
        tenant_id = _current_tenant_id()
        await self._lock_student(tenant_id, user_id)
        event = await self._event(tenant_id, user_id, event_id)
        topics = await self._contribution_topics(event.id) | {t for t, _ in observations}
        await self.db.execute(
            delete(LearningMemoryContribution).where(
                LearningMemoryContribution.event_row_id == event.id
            )
        )
        await self._add_contributions(tenant_id, user_id, event.id, observations)
        event.changed_at = datetime.now(UTC)
        await self.db.flush()
        await self._recompute(tenant_id, user_id, topics)
        await self.db.commit()
        return await self.query(user_id, sorted(topics))

    async def delete_event(self, user_id: str, event_id: str) -> None:
        """Remove one event's contributions. The event row stays, so a late retry
        can't bring them back."""
        _require_gate_open("write")
        tenant_id = _current_tenant_id()
        await self._lock_student(tenant_id, user_id)
        event = await self._event(tenant_id, user_id, event_id)
        topics = await self._contribution_topics(event.id)
        await self.db.execute(
            delete(LearningMemoryContribution).where(
                LearningMemoryContribution.event_row_id == event.id
            )
        )
        event.changed_at = datetime.now(UTC)
        await self.db.flush()
        await self._recompute(tenant_id, user_id, topics)
        await self.db.commit()

    async def recompute(self, user_id: str, topic: str) -> None:
        """Rebuild one topic's aggregate from its contributions."""
        _require_gate_open("write")
        tenant_id = _current_tenant_id()
        await self._lock_student(tenant_id, user_id)
        await self._recompute(tenant_id, user_id, {topic})
        await self.db.commit()

    async def _lock_student(self, tenant_id: str, user_id: str) -> None:
        # One change at a time per student, until the transaction ends. So
        # contributions are numbered in the order they applied, and a recompute
        # gets the same answer even though clamping depends on order.
        key = f"learning-memory:{tenant_id}:{user_id}"
        await self.db.execute(select(func.pg_advisory_xact_lock(func.hashtextextended(key, 0))))

    async def _insert_event(
        self,
        tenant_id: str,
        user_id: str,
        event_id: str | None,
        source: tuple[str, str] | None,
        now: datetime,
    ) -> int | None:
        """The new row's id, or None if this event was already reported."""
        statement = (
            insert(LearningMemoryEvent)
            .values(
                tenant_id=tenant_id,
                user_id=user_id,
                event_id=event_id or f"unidentified-{uuid4()}",
                source_kind=source[0] if source else None,
                source_id=source[1] if source else None,
                received_at=now,
            )
            .on_conflict_do_nothing(constraint="uq_learning_memory_events_tenant_event")
            .returning(LearningMemoryEvent.id)
        )
        return (await self.db.execute(statement)).scalar_one_or_none()

    async def _check_repeat(
        self, tenant_id: str, user_id: str, event_id: str, observations: list[tuple[str, bool]]
    ) -> None:
        """A repeat must be the same event. Another student's or a different
        body is a caller's bug, and silently dropping it would lose data."""
        event = (
            await self.db.execute(
                select(LearningMemoryEvent).where(
                    LearningMemoryEvent.tenant_id == tenant_id,
                    LearningMemoryEvent.event_id == event_id,
                )
            )
        ).scalar_one()
        if event.user_id != user_id:
            raise LearningMemoryEventConflict
        if event.changed_at is not None:
            return  # deliberately changed since: the retry has nothing to say
        stored = await self.db.execute(
            select(LearningMemoryContribution.topic, LearningMemoryContribution.correct)
            .where(LearningMemoryContribution.event_row_id == event.id)
            .order_by(LearningMemoryContribution.position)
        )
        if [tuple(row) for row in stored.all()] != list(observations):
            raise LearningMemoryEventConflict

    async def _event(self, tenant_id: str, user_id: str, event_id: str) -> LearningMemoryEvent:
        event = (
            await self.db.execute(
                select(LearningMemoryEvent).where(
                    LearningMemoryEvent.tenant_id == tenant_id,
                    LearningMemoryEvent.user_id == user_id,
                    LearningMemoryEvent.event_id == event_id,
                )
            )
        ).scalar_one_or_none()
        if event is None:
            raise LookupError("learning memory event not found")
        return event

    async def _contribution_topics(self, event_row_id: int) -> set[str]:
        rows = await self.db.execute(
            select(LearningMemoryContribution.topic).where(
                LearningMemoryContribution.event_row_id == event_row_id
            )
        )
        return set(rows.scalars().all())

    async def _add_contributions(
        self, tenant_id: str, user_id: str, event_row_id: int, observations: list[tuple[str, bool]]
    ) -> None:
        self.db.add_all(
            LearningMemoryContribution(
                event_row_id=event_row_id,
                tenant_id=tenant_id,
                user_id=user_id,
                topic=topic,
                correct=correct,
                position=position,
            )
            for position, (topic, correct) in enumerate(observations)
        )
        await self.db.flush()

    async def _recompute(self, tenant_id: str, user_id: str, topics: set[str]) -> None:
        # Before any change, so a refusal leaves every topic as it was. The
        # caller's uncommitted work rolls back with the session.
        incomplete = await self.db.execute(
            select(LearningMemory.topic).where(
                LearningMemory.tenant_id == tenant_id,
                LearningMemory.user_id == user_id,
                LearningMemory.topic.in_(topics),
                LearningMemory.history_complete.is_(False),
            )
        )
        if incomplete.first() is not None:
            await self.db.rollback()
            raise LearningMemoryHistoryIncomplete
        for topic in topics:
            rows = (
                await self.db.execute(
                    select(LearningMemoryContribution.correct, LearningMemoryEvent.received_at)
                    .join(
                        LearningMemoryEvent,
                        LearningMemoryContribution.event_row_id == LearningMemoryEvent.id,
                    )
                    .where(
                        LearningMemoryContribution.tenant_id == tenant_id,
                        LearningMemoryContribution.user_id == user_id,
                        LearningMemoryContribution.topic == topic,
                    )
                    .order_by(LearningMemoryEvent.id, LearningMemoryContribution.position)
                )
            ).all()
            if not rows:
                await self.db.execute(
                    delete(LearningMemory).where(
                        LearningMemory.tenant_id == tenant_id,
                        LearningMemory.user_id == user_id,
                        LearningMemory.topic == topic,
                    )
                )
                continue
            mastery = fold_mastery([correct for correct, _ in rows])
            stamp = max(received for _, received in rows).astimezone(UTC).isoformat()
            statement = insert(LearningMemory).values(
                tenant_id=tenant_id,
                user_id=user_id,
                topic=topic,
                mastery_score=mastery,
                attempts=len(rows),
                last_interaction=stamp,
                history_complete=True,
            )
            await self.db.execute(
                statement.on_conflict_do_update(
                    constraint="uq_learning_memory_tenant_user_topic",
                    set_={
                        "mastery_score": mastery,
                        "attempts": len(rows),
                        "last_interaction": stamp,
                    },
                )
            )
