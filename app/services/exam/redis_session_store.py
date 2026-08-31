import json
import time

from app.core.redis import redis_client


class RedisSessionStore:
    def _key(self, session_id: str) -> str:
        return f"exam_session:{session_id}"

    async def create_session(self, session_id: str, data: dict):
        payload = {
            "user_id": data.get("user_id"),
            "answers": data.get("answers", {}),
            "start_time": data.get("start_time", time.time()),
            "status": data.get("status", "active"),
            "last_activity": data.get("last_activity", time.time()),
        }

        await redis_client.set(
            self._key(session_id),
            json.dumps(payload),
        )

    async def get_session(self, session_id: str):
        data = await redis_client.get(self._key(session_id))
        return json.loads(data) if data else None

    async def update_answer(self, session_id: str, question_id: str, answer: str):
        session = await self.get_session(session_id)

        if not session:
            return None

        session["answers"][question_id] = answer
        session["last_activity"] = time.time()

        await redis_client.set(
            self._key(session_id),
            json.dumps(session),
        )

        return session
