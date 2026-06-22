import pytest
from app.db.session import verify_connection


async def test_db_connection():
    await verify_connection()
    assert True
