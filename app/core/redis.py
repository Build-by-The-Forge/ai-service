"""Shared Redis access for the task queue, idempotency and readiness.

An asyncio connection pool belongs to the event loop that opened it, so a
single module-global client breaks as soon as code runs in a second loop
(the error then surfaces far from its cause). There is one client per
running event loop instead; the URL's database number is honoured.
"""

import asyncio
import weakref

import redis.asyncio as redis

from app.core.config import settings

redis_url = settings.REDIS_URL or f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"

_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, redis.Redis]" = (
    weakref.WeakKeyDictionary()
)


def _new_client() -> redis.Redis:
    return redis.Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_timeout=1.0,
        socket_connect_timeout=1.0,
    )


def get_redis_client() -> redis.Redis:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return _new_client()
    client = _clients.get(loop)
    if client is None:
        client = _clients[loop] = _new_client()
    return client
