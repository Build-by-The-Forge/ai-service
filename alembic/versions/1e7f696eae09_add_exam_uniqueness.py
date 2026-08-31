"""add exam uniqueness

Revision ID: 1e7f696eae09
Revises:
Create Date: 2026-06-22 17:20:32.092338
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1e7f696eae09"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_exam_title_school",
        "exams",
        ["title", "school_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_exam_title_school",
        "exams",
        type_="unique",
    )
