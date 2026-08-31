import redis.asyncio as redis

from app.core.config import settings

redis_url = settings.REDIS_URL or f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"

redis_client = redis.Redis.from_url(
    redis_url,
    db=0,
    decode_responses=True,
    socket_timeout=1.0,
    socket_connect_timeout=1.0,
)


def get_redis_client() -> redis.Redis:
    return redis_client
