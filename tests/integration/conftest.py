import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import app.db.session as session_module
from app.db.session import initialize_database, reset_engine
from app.main import create_app


@pytest_asyncio.fixture(autouse=True)
async def engine_disposed_on_the_test_loop():
    """Fail any integration test that leaves the module-global engine open.

    Each test runs on its own event loop, and pooled asyncpg connections
    belong to the loop that opened them. An engine left open is disposed by a
    later test on a different loop, where closing those connections fails
    with "Event loop is closed".
    """
    yield
    if session_module.engine is not None:
        await reset_engine()
        pytest.fail(
            "the test left app.db.session.engine open; call reset_engine() "
            "before it ends (the app fixture does this)"
        )


@pytest_asyncio.fixture
async def app():
    try:
        # Attempt to initialize database; skip integration tests if unavailable.
        await initialize_database()
    except Exception as exc:  # pragma: no cover - integration guard
        import pytest

        pytest.skip(f"Database not available for integration tests: {exc}")

    app = create_app()
    try:
        yield app
    finally:
        await reset_engine()


@pytest_asyncio.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


def requires_db():
    return bool(os.environ.get("DATABASE_URL"))


def requires_redis():
    return bool(os.environ.get("REDIS_URL"))
