from typing import AsyncGenerator

from app.db.session import get_async_session_local


async def get_db() -> AsyncGenerator:
    session_factory = get_async_session_local()
    async with session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_settings():
    from app.core.config import settings
    return settings
