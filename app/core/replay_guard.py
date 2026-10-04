"""Request-id replay guard for inbound signed service calls (design §A.5 step 8).

Keys are ``svcauth:<receiver>:<sender>:<request-id>``, set with NX and a
180 s expiry — longer than the ±60 s acceptance window, so an id can never
be reused while its timestamp is still fresh. There is no in-process
fallback: a per-replica memory would let a request be replayed against
another replica, so an unreachable Redis means the request is refused.
"""

from __future__ import annotations

import redis.asyncio as redis_asyncio
from redis.exceptions import RedisError

REPLAY_WINDOW_SECONDS = 180


class ReplayGuardUnavailableError(Exception):
    """Redis did not answer; the caller must fail closed (503)."""


def replay_key(receiver: str, sender: str, request_id: str) -> str:
    return f"svcauth:{receiver}:{sender}:{request_id}"


class RedisReplayGuard:
    def __init__(self, client: redis_asyncio.Redis) -> None:
        self._client = client

    @classmethod
    def connect(cls, url: str, *, timeout: float = 1.0) -> RedisReplayGuard:
        return cls(
            redis_asyncio.Redis.from_url(
                url,
                decode_responses=True,
                socket_timeout=timeout,
                socket_connect_timeout=timeout,
            )
        )

    async def first_use(self, receiver: str, sender: str, request_id: str) -> bool:
        """Record the id; True only if it had not been seen before."""
        key = replay_key(receiver, sender, request_id)
        try:
            stored = await self._client.set(key, "1", nx=True, ex=REPLAY_WINDOW_SECONDS)
        except (RedisError, OSError) as exc:
            raise ReplayGuardUnavailableError(type(exc).__name__) from exc
        return stored is True

    async def close(self) -> None:
        await self._client.close()  # redis-py 5.0.0 has no aclose()
