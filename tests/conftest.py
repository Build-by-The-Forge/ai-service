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


@pytest.fixture(autouse=True)
def in_memory_retrieval():
    """Tests retrieve from an in-memory engine (tests/rag_support.py); the
    app's default is the durable one, which tests that need it build."""
    from app.services.rag.retrieval import set_retrieval_engine
    from tests.rag_support import in_memory_engine

    engine = in_memory_engine()
    set_retrieval_engine(engine)
    yield engine
    set_retrieval_engine(None)
