from app.db.session import AsyncSessionLocal
from typing import AsyncGenerator


async def get_db() -> AsyncGenerator:
    async with AsyncSessionLocal() as session:
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
