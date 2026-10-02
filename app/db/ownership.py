"""Which tables the AI service owns in the database it shares with the Build backend.

The backend owns the platform schema (users, tenants, questions, assessments, ...)
and migrates it with its own Alembic chain. The AI service may only create or
migrate the tables listed here.

`Base.metadata` still contains legacy models (users, schools, exams,
exam_sessions, answer_sheets, questions) that pre-date the shared database and
are kept only for existing runtime code until the identity work replaces them.
They are deliberately absent from this set, so Alembic never creates, alters,
or proposes dropping them -- nor any backend table it reflects from the database.
"""

from typing import Any

AI_OWNED_TABLES = frozenset({"learning_memory", "session_memory"})


def include_object(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    """Alembic `include_object` hook restricting migrations to AI-owned tables."""
    if type_ == "table":
        return name in AI_OWNED_TABLES
    table = getattr(obj, "table", None)
    return table is not None and table.name in AI_OWNED_TABLES
