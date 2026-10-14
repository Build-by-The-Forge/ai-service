import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_db_connection():
    from app.db.session import get_async_session_local, reset_engine

    session_factory = get_async_session_local()
    try:
        async with session_factory() as db:
            result = await db.execute(text("SELECT 1"))
            assert result.scalar() == 1
    finally:
        # The engine is module-global but each test has its own event loop:
        # dispose of it on this loop, or a later test has to close a pooled
        # connection whose loop is already closed.
        await reset_engine()
