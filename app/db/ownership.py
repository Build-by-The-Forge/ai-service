"""Which tables the AI service owns in the database it shares with the Build backend.

The backend owns the platform schema (users, tenants, questions, assessments, ...)
and migrates it with its own Alembic chain. The AI service may only create or
migrate the tables listed here. `Base.metadata` holds only these tables since
M4 removed the legacy models; this filter still keeps Alembic from touching
any backend table it reflects from the shared database.
"""

from typing import Any

AI_OWNED_TABLES = frozenset(
    {"document_chunks", "ingestion_fences", "learning_memory", "session_memory"}
)


def include_object(
    obj: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    """Alembic `include_object` hook restricting migrations to AI-owned tables."""
    if type_ == "table":
        return name in AI_OWNED_TABLES
    table = getattr(obj, "table", None)
    return table is not None and table.name in AI_OWNED_TABLES
