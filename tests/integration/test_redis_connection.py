import pytest
from redis.asyncio import Redis


@pytest.mark.asyncio
async def test_redis_ping():
    client = Redis(host="localhost", port=6379, decode_responses=True)

    try:
        assert await client.ping() is True
    finally:
        await client.close()
