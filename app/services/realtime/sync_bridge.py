from app.services.realtime.pubsub import RedisPubSub


class SyncBridge:

    def __init__(self):
        self.pubsub = RedisPubSub()

    async def broadcast_answer(self, session_id: str, data: dict):
        await self.pubsub.publish(
            channel=f"exam:{session_id}",
            message={
                "event": "answer_sync",
                "source": data.get("source"),
                "data": data,
            },
        )
