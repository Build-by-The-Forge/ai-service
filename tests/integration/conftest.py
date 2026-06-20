import os
import asyncio
import pytest
from httpx import AsyncClient

from app.main import create_app
from app.core.config import settings


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def app():
    app = create_app()
    yield app


@pytest.fixture
async def client(app):
    async with AsyncClient(app=app, base_url="http://testserver") as ac:
        yield ac


def requires_db():
    return bool(os.environ.get("DATABASE_URL"))


def requires_redis():
    return bool(os.environ.get("REDIS_URL"))
