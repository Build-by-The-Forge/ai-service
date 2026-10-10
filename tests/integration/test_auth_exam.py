import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.db.models.user import User
from app.main import create_app
from app.db.session import get_async_session_local
from tests.integration.conftest import seed_user


@pytest_asyncio.fixture
async def app(app):
    # The legacy routes are unmounted by default (M3-10) and deleted in M4.
    return create_app(legacy_user_routes=True)


@pytest_asyncio.fixture
async def seeded_user():
    user, created = await seed_user()

    try:
        yield user
    finally:
        if created:
            session_factory = get_async_session_local()
            async with session_factory() as db:
                await db.delete(user)
                await db.commit()


@pytest.mark.asyncio
async def test_login_returns_access_token(client: AsyncClient, seeded_user: User):
    response = await client.post(
        "/auth/login",
        params={"email": seeded_user.email, "password": "changeme"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["access_token"]
