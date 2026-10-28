from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class IngestionFence(Base):
    """Which attempt may currently write an asset's index (P3-6 §7.5).

    Raised when AI-service accepts a dispatch, before the task is queued,
    and only ever raised: ``(run_number, attempt)`` never goes down. An
    index write must equal it, checked under this row's lock, so only the
    attempt the backend most recently dispatched can write the asset.
    """

    __tablename__ = "ingestion_fences"

    # Backend `tenants.id` and content asset id values; not foreign keys.
    tenant_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_number: Mapped[int] = mapped_column(Integer)
    attempt: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
