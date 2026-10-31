from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    false,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class LearningMemory(Base):
    __tablename__ = "learning_memory"
    # One row per learner and topic within a tenant (memory upserts, M5).
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "user_id", "topic", name="uq_learning_memory_tenant_user_topic"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Backend `tenants.id` value. Deliberately not a foreign key: the backend
    # owns `tenants` and this service owns this table; tenant isolation is
    # enforced in the application layer.
    tenant_id: Mapped[str] = mapped_column(String(36), index=True)
    user_id: Mapped[str] = mapped_column(String(100), index=True)
    topic: Mapped[str] = mapped_column(String(255), index=True)
    mastery_score: Mapped[float] = mapped_column(Float, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_interaction: Mapped[str] = mapped_column(String(50))
    # False for a row from before contributions existed. Its history is gone and
    # is never manufactured, so it can't be recomputed: a recompute would lose it.
    history_complete: Mapped[bool] = mapped_column(Boolean, default=True, server_default=false())


class LearningMemoryEvent(Base):
    """One reported event, kept as long as the memory it fed. It's what makes a
    repeat a no-op, so it survives its contributions being deleted or revised:
    a late retry must not bring them back. Holds IDs only, no answers or documents.
    """

    __tablename__ = "learning_memory_events"
    __table_args__ = (
        UniqueConstraint("tenant_id", "event_id", name="uq_learning_memory_events_tenant_event"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(36), index=True)
    user_id: Mapped[str] = mapped_column(String(100), index=True)
    event_id: Mapped[str] = mapped_column(String(255))
    # Generic, so document sources can be added later without a redesign.
    source_kind: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # Set once its contributions are changed or removed on purpose.
    changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class LearningMemoryContribution(Base):
    """One observation from one event. The aggregate can be recomputed from these."""

    __tablename__ = "learning_memory_contributions"
    __table_args__ = (
        UniqueConstraint(
            "event_row_id", "position", name="uq_learning_memory_contributions_event_position"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_row_id: Mapped[int] = mapped_column(
        ForeignKey("learning_memory_events.id", ondelete="CASCADE"), index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(36), index=True)
    user_id: Mapped[str] = mapped_column(String(100), index=True)
    topic: Mapped[str] = mapped_column(String(255), index=True)
    correct: Mapped[bool] = mapped_column(Boolean)
    position: Mapped[int] = mapped_column(Integer)
