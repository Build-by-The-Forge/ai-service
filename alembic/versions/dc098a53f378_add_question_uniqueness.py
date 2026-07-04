"""add question uniqueness

Revision ID: dc098a53f378
Revises: 1e7f696eae09
Create Date: 2026-06-22 17:50:00.000000
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "dc098a53f378"
down_revision = "1e7f696eae09"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_question_school_course_topic",
        "questions",
        ["school_id", "course", "topic"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_question_school_course_topic",
        "questions",
        type_="unique",
    )
