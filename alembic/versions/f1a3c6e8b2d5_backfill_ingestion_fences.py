"""Backfill ingestion_fences from the existing index (P3-6 §11)

Every asset already in document_chunks gets a fence at the (run, attempt)
its chunks were written by: the highest run, then the highest attempt within
it. A fence that already exists is kept: a dispatch raised it, and it is
never lower than the chunks it admitted. Without this, an asset indexed
before fencing has no fence row until its next dispatch, so nothing could
write it in between, but the design records its current attempt instead.

Revision ID: f1a3c6e8b2d5
Revises: e5b7d2c9a184
Create Date: 2026-10-28 18:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f1a3c6e8b2d5"
down_revision: Union[str, Sequence[str], None] = "e5b7d2c9a184"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

BACKFILL = """
INSERT INTO ingestion_fences (tenant_id, asset_id, run_number, attempt)
SELECT DISTINCT ON (tenant_id, asset_id) tenant_id, asset_id, run_number, attempt
FROM document_chunks
ORDER BY tenant_id, asset_id, run_number DESC, attempt DESC
ON CONFLICT (tenant_id, asset_id) DO NOTHING
"""


def upgrade() -> None:
    op.execute(sa.text(BACKFILL))


def downgrade() -> None:
    # Fences are only ever raised; removing backfilled ones would lower them.
    pass
