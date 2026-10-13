import os

# Tests run as the "testing" environment: service secrets that are not set
# explicitly are generated per process (app.core.config). Must run before
# app.core.config is first imported.
os.environ.setdefault("APP_ENVIRONMENT", "testing")


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def in_memory_task_records():
    """Unit tests keep task records in memory; tests that exercise the real
    Redis store install their own."""
    from app.services.tasks.task_records import InMemoryTaskRecordStore, set_task_record_store

    store = InMemoryTaskRecordStore()
    set_task_record_store(store)
    yield store
    set_task_record_store(None)
