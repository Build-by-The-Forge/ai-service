import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.db.base import Base

DATABASE_URL = settings.DATABASE_URL

engine = None
AsyncSessionLocal = None


def _build_engine():
    global engine, AsyncSessionLocal
    if engine is None:
        engine = create_async_engine(
            DATABASE_URL,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        AsyncSessionLocal = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return engine


def get_async_session_local():
    _build_engine()
    return AsyncSessionLocal


def get_engine():
    """Return the active SQLAlchemy async engine, building it if necessary."""
    return _build_engine()


async def initialize_database() -> None:
    eng = _build_engine()
    # Retry loop: database containers can be starting when tests begin.
    max_attempts = 10
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            async with eng.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            break
        except Exception as exc:
            logging.debug("initialize_database attempt %d failed: %s", attempt, exc)
            if attempt == max_attempts:
                raise
            await asyncio.sleep(delay)
            delay = min(5.0, delay * 2)

    # Keep the engine alive for the application lifespan so request-scoped
    # sessions can reuse the same pool instead of rebuilding it on each init.
    # Disposal happens explicitly during shutdown/reset.


async def verify_connection() -> None:
    eng = _build_engine()
    async with eng.begin() as conn:
        await conn.execute(text("SELECT 1"))


async def reset_engine() -> None:
    global engine, AsyncSessionLocal
    if engine is not None:
        try:
            await engine.dispose()
        except Exception as exc:
            logging.warning("Failed to dispose SQLAlchemy engine cleanly: %s", exc)
    engine = None
    AsyncSessionLocal = None
