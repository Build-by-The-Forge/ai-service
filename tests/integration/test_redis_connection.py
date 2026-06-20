import pytest
import os

from app.core.redis import redis_client


@pytest.mark.asyncio
async def test_redis_ping_or_skip():
    if not os.environ.get("REDIS_URL"):
        pytest.skip("REDIS_URL not set; skipping Redis integration test")

    res = await redis_client.ping()
    assert res is True
