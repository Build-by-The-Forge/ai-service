"""ingestion_fences and document_chunks.attempt: the index fence (P3-6 §7.5)

ingestion_fences(tenant_id, asset_id) -> (run_number, attempt) records which
attempt may currently write each asset's index. It is raised when AI-service
accepts a dispatch and only ever rises; an index write must equal it.

document_chunks.attempt records the attempt that wrote each row, for
provenance. Existing rows were written before fencing, by attempt 1 of their
run as far as the index can tell, so the column is backfilled with 1.

Revision ID: e5b7d2c9a184
Revises: c3e81f4a9d27
Create Date: 2026-10-28 12:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e5b7d2c9a184"
down_revision: Union[str, Sequence[str], None] = "c3e81f4a9d27"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ingestion_fences",
        sa.Column("tenant_id", sa.String(36), primary_key=True),
        sa.Column("asset_id", sa.String(64), primary_key=True),
        sa.Column("run_number", sa.Integer(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    # The server default backfills existing rows in the same statement.
    op.add_column(
        "document_chunks",
        sa.Column("attempt", sa.Integer(), server_default="1", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("document_chunks", "attempt")
    op.drop_table("ingestion_fences")
