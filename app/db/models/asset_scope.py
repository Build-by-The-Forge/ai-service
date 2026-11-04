from datetime import datetime

from sqlalchemy import BigInteger, CheckConstraint, DateTime, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class AssetScope(Base):
    """A document's knowledge scope (M3-D), one row per document.

    Retrieval joins chunks to this row, so a document's scope is never
    copied onto its chunks: a reprocess, or chunks a job in flight writes
    late, take the document's current scope. A document without a row is
    ``unknown``. ``version`` is the backend's, and a row changes only to a
    higher one, so a replayed or late assignment changes nothing.
    """

    __tablename__ = "asset_scopes"

    # Backend `tenants.id` and content asset id values; not foreign keys.
    tenant_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scope: Mapped[str] = mapped_column(String(16))
    version: Mapped[int] = mapped_column(BigInteger)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "scope IN ('institutional', 'personal', 'unknown')", name="ck_asset_scopes_scope"
        ),
        CheckConstraint("version > 0", name="ck_asset_scopes_version"),
        # An assignment checks that no other tenant holds the asset.
        Index("ix_asset_scopes_asset", "asset_id"),
    )
