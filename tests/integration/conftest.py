import os

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.db.session import initialize_database, reset_engine
from app.main import create_app


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
