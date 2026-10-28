from datetime import datetime

from sqlalchemy import Computed, DateTime, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.types import Vector

# all-MiniLM-L6-v2 (D3). A model with another dimension needs a migration.
EMBEDDING_DIMENSION = 384


class DocumentChunk(Base):
    """One indexed chunk of a document asset (P3-4).

    Rows are always read and written with the verified tenant first; the
    tenant is the backend's `tenants.id` value, not a foreign key (the
    backend owns that table). An asset's chunks are replaced as a whole, in
    one transaction, when it is re-indexed.
    """

    __tablename__ = "document_chunks"

    # Deterministic (tenant, asset, run, index) uuid5 from the chunker.
    chunk_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(36))
    asset_id: Mapped[str] = mapped_column(String(64))
    run_number: Mapped[int] = mapped_column(Integer)
    # The attempt that wrote the row, for provenance; ingestion_fences decides
    # which attempt may write (P3-6 §7.5).
    attempt: Mapped[int] = mapped_column(Integer, server_default="1")
    chunk_index: Mapped[int] = mapped_column(Integer)
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSION))
    embedding_model: Mapped[str] = mapped_column(String(100))
    # Postgres full-text search: the keyword side of hybrid retrieval.
    content_tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True)
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_document_chunks_tenant_asset", "tenant_id", "asset_id"),
        Index("ix_document_chunks_tsv", "content_tsv", postgresql_using="gin"),
        Index(
            "ix_document_chunks_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
