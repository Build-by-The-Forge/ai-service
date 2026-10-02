from sqlalchemy import Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class LearningMemory(Base):
    __tablename__ = "learning_memory"

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
