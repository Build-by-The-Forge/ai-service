import os

import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from passlib.hash import pbkdf2_sha256
from sqlalchemy import select

from app.db.models.user import User
from app.db.session import get_async_session_local, initialize_database, reset_engine
from app.main import create_app


@pytest_asyncio.fixture
async def app():
    await initialize_database()
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


async def seed_user(
    email: str = "student@demo.edu",
    password: str = "changeme",
    school_id: int = 1,
    role: str = "student",
) -> tuple[User, bool]:
    session_factory = get_async_session_local()
    async with session_factory() as db:
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        created = False

        if user is None:
            user = User(
                email=email,
                hashed_password=pbkdf2_sha256.hash(password),
                school_id=school_id,
                role=role,
            )
            db.add(user)
            created = True
        else:
            user.hashed_password = pbkdf2_sha256.hash(password)
            user.school_id = school_id
            user.role = role

        await db.commit()
        await db.refresh(user)
        return user, created


def requires_db():
    return bool(os.environ.get("DATABASE_URL"))


def requires_redis():
    return bool(os.environ.get("REDIS_URL"))
