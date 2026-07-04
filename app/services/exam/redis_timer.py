import time
from app.core.redis import redis_client


class RedisExamTimer:
    def _key(self, session_id: str) -> str:
        return f"exam_timer:{session_id}"

    async def start(self, session_id: str, duration: int):
        await redis_client.set(
            self._key(session_id),
            time.time() + (duration * 60),
        )

    async def remaining(self, session_id: str) -> float:
        end_time = await redis_client.get(self._key(session_id))

        if not end_time:
            return 0.0

        return max(0.0, float(end_time) - time.time())

    async def is_expired(self, session_id: str) -> bool:
        return await self.remaining(session_id) <= 0.0
