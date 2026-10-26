"""document_chunks: the durable hybrid-retrieval index (P3-4)

One row per chunk of an indexed document asset: the chunk text with its
page span, a 384-d embedding (all-MiniLM-L6-v2, D3) and a generated
full-text vector, with the model id stored per row.

The `vector` extension is provisioned, not migrated: the AI owner role
cannot create extensions (db-init does it in compose, the dashboard on
Supabase), so the upgrade only checks that it exists and refuses otherwise.

Revision ID: c3e81f4a9d27
Revises: a7c2e5d10b41
Create Date: 2026-10-26 18:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import TSVECTOR

from app.db.types import Vector

revision: str = "c3e81f4a9d27"
down_revision: Union[str, Sequence[str], None] = "a7c2e5d10b41"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "document_chunks"


def upgrade() -> None:
    present = (
        op.get_bind()
        .execute(sa.text("SELECT 1 FROM pg_extension WHERE extname = 'vector'"))
        .scalar()
    )
    if not present:
        raise RuntimeError(
            "the pgvector extension is not enabled in this database; a superuser must run "
            "CREATE EXTENSION vector (db-init does in compose; enable it in the Supabase "
            "dashboard) before upgrading"
        )
    op.create_table(
        TABLE,
        sa.Column("chunk_id", sa.String(36), primary_key=True),
        sa.Column("tenant_id", sa.String(36), nullable=False),
        sa.Column("asset_id", sa.String(64), nullable=False),
        sa.Column("run_number", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("embedding_model", sa.String(100), nullable=False),
        sa.Column(
            "content_tsv",
            TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    # Tenant first: every read and every replace filters on the tenant.
    op.create_index("ix_document_chunks_tenant_asset", TABLE, ["tenant_id", "asset_id"])
    op.create_index("ix_document_chunks_tsv", TABLE, ["content_tsv"], postgresql_using="gin")
    op.create_index(
        "ix_document_chunks_embedding",
        TABLE,
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_table(TABLE)
