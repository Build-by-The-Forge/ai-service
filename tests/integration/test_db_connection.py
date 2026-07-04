import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_db_connection():
    from app.db.session import get_async_session_local

    session_factory = get_async_session_local()
    async with session_factory() as db:
        result = await db.execute(text("SELECT 1"))
        assert result.scalar() == 1
