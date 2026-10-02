from types import SimpleNamespace

import pytest

from app.db.ownership import AI_OWNED_TABLES, include_object


@pytest.mark.parametrize("table", sorted(AI_OWNED_TABLES))
def test_ai_owned_tables_are_included(table):
    assert include_object(None, table, "table", False, None)


@pytest.mark.parametrize(
    "table",
    [
        # backend-owned tables reflected from the shared database
        "users",
        "tenants",
        "questions",
        "assessments",
        "alembic_version",
        # legacy AI models that must never be migrated
        "schools",
        "exams",
        "exam_sessions",
        "answer_sheets",
    ],
)
@pytest.mark.parametrize("reflected", [True, False])
def test_other_tables_are_excluded(table, reflected):
    assert not include_object(None, table, "table", reflected, None)


def test_child_objects_follow_their_table():
    owned = SimpleNamespace(table=SimpleNamespace(name="learning_memory"))
    backend = SimpleNamespace(table=SimpleNamespace(name="users"))

    assert include_object(owned, "ix_learning_memory_topic", "index", False, None)
    assert not include_object(backend, "ix_users_email", "index", True, None)
    assert not include_object(backend, "email", "column", True, None)


def test_object_without_table_is_excluded():
    assert not include_object(SimpleNamespace(), "orphan", "index", True, None)
