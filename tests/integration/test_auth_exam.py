import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import create_app
from app.db.session import AsyncSessionLocal
from app.db.models.user import User
from passlib.hash import pbkdf2_sha256


@pytest.fixture(scope="session")
async def app():
    app = create_app()
    yield app


@pytest.fixture
async def client(app):
    async with AsyncClient(app=app, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture
async def user_fixture():
    return {
        "email": "student@demo.edu",
        "password": "changeme",
        "school_id": 1,
        "role": "student",
    }


@pytest.mark.asyncio
async def test_login_returns_access_token(client: AsyncClient):
    response = await client.post("/auth/login", json={"email": "student@demo.edu", "password": "changeme"})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["access_token"]
