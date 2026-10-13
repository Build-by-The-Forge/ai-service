"""learning_memory: one row per tenant, user and topic

Memory upserts (M5) rely on this key for INSERT ... ON CONFLICT. Existing
duplicate rows are not merged or deleted: the upgrade refuses and reports
them, so no learner history is silently lost.

Revision ID: a7c2e5d10b41
Revises: 3409868eb53a
Create Date: 2026-10-10 18:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a7c2e5d10b41"
down_revision: Union[str, Sequence[str], None] = "3409868eb53a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

CONSTRAINT = "uq_learning_memory_tenant_user_topic"


def upgrade() -> None:
    duplicates = (
        op.get_bind()
        .execute(
            sa.text(
                "SELECT tenant_id, user_id, topic, count(*) AS n FROM learning_memory"
                " GROUP BY tenant_id, user_id, topic HAVING count(*) > 1 LIMIT 10"
            )
        )
        .fetchall()
    )
    if duplicates:
        listed = "; ".join(f"{r.tenant_id}/{r.user_id}/{r.topic} x{r.n}" for r in duplicates)
        raise RuntimeError(
            "learning_memory has duplicate (tenant_id, user_id, topic) rows; merge them "
            f"before upgrading (first {len(duplicates)}: {listed})"
        )
    op.create_unique_constraint(CONSTRAINT, "learning_memory", ["tenant_id", "user_id", "topic"])


def downgrade() -> None:
    op.drop_constraint(CONSTRAINT, "learning_memory", type_="unique")
