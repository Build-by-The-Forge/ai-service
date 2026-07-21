import pytest
from redis.asyncio import Redis


@pytest.mark.asyncio
async def test_redis_ping():
    client = Redis(host="localhost", port=6379, decode_responses=True)

    try:
        try:
            assert await client.ping() is True
        except Exception as exc:  # pragma: no cover - redis may be absent in some envs
            import pytest

            pytest.skip(f"Redis unavailable: {exc}")
    finally:
        await client.close()
