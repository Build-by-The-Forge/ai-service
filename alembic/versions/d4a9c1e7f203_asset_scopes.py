"""asset_scopes: each document's knowledge scope (M3-D)

asset_scopes(tenant_id, asset_id) -> (scope, version) holds the scope the
backend last assigned each document, and retrieval returns only chunks
whose document is `institutional`. No rows are created: every document
indexed before this revision has none, so it is `unknown` and out of
retrieval until the backend assigns it a scope. Nothing is assumed.

ingestion_fences gains an index on asset_id, so an assignment can check
that no other tenant holds the asset without scanning the table.

Revision ID: d4a9c1e7f203
Revises: b8d4f2a6c1e3
Create Date: 2026-11-04 12:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d4a9c1e7f203"
down_revision: Union[str, Sequence[str], None] = "b8d4f2a6c1e3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "asset_scopes",
        sa.Column("tenant_id", sa.String(36), primary_key=True),
        sa.Column("asset_id", sa.String(64), primary_key=True),
        sa.Column("scope", sa.String(16), nullable=False),
        sa.Column("version", sa.BigInteger(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "scope IN ('institutional', 'personal', 'unknown')", name="ck_asset_scopes_scope"
        ),
        sa.CheckConstraint("version > 0", name="ck_asset_scopes_version"),
    )
    op.create_index("ix_asset_scopes_asset", "asset_scopes", ["asset_id"])
    op.create_index("ix_ingestion_fences_asset", "ingestion_fences", ["asset_id"])


def downgrade() -> None:
    op.drop_index("ix_ingestion_fences_asset", table_name="ingestion_fences")
    op.drop_table("asset_scopes")
