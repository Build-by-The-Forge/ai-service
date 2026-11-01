"""learning_memory_events and learning_memory_contributions

One row per reported event, and one per observation, so a repeat changes
nothing and the aggregate can be recomputed. Existing learning_memory rows have
no source events to rebuild from, and no history is manufactured: they're marked
history_complete = false, and a recompute refuses them.

Revision ID: b8d4f2a6c1e3
Revises: f1a3c6e8b2d5
Create Date: 2026-10-31 20:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b8d4f2a6c1e3"
down_revision: Union[str, Sequence[str], None] = "f1a3c6e8b2d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # The server default marks every existing row; new rows set true themselves.
    op.add_column(
        "learning_memory",
        sa.Column("history_complete", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_table(
        "learning_memory_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("tenant_id", sa.String(36), nullable=False),
        sa.Column("user_id", sa.String(100), nullable=False),
        sa.Column("event_id", sa.String(255), nullable=False),
        sa.Column("source_kind", sa.String(50), nullable=True),
        sa.Column("source_id", sa.String(255), nullable=True),
        sa.Column("payload_fingerprint", sa.String(64), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("changed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("tenant_id", "event_id", name="uq_learning_memory_events_tenant_event"),
    )
    op.create_index("ix_learning_memory_events_tenant_id", "learning_memory_events", ["tenant_id"])
    op.create_index("ix_learning_memory_events_user_id", "learning_memory_events", ["user_id"])

    op.create_table(
        "learning_memory_contributions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "event_row_id",
            sa.Integer(),
            sa.ForeignKey("learning_memory_events.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tenant_id", sa.String(36), nullable=False),
        sa.Column("user_id", sa.String(100), nullable=False),
        sa.Column("topic", sa.String(255), nullable=False),
        sa.Column("correct", sa.Boolean(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.UniqueConstraint(
            "event_row_id", "position", name="uq_learning_memory_contributions_event_position"
        ),
    )
    for column in ("event_row_id", "tenant_id", "user_id", "topic"):
        op.create_index(
            f"ix_learning_memory_contributions_{column}", "learning_memory_contributions", [column]
        )


def downgrade() -> None:
    op.drop_table("learning_memory_contributions")
    op.drop_table("learning_memory_events")
    op.drop_column("learning_memory", "history_complete")
