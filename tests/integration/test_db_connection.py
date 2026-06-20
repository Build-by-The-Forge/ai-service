import pytest
import os

from app.db.session import test_connection


@pytest.mark.asyncio
async def test_db_connection_or_skip():
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set; skipping DB integration test")

    await test_connection()
    assert True
