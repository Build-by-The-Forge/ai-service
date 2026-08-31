import json

from app.core.redis import redis_client


class RedisPubSub:
    async def publish(self, channel: str, message: dict):
        await redis_client.publish(channel, json.dumps(message))

    async def subscribe(self, channel: str, handler):
        pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
        await pubsub.subscribe(channel)

        async for msg in pubsub.listen():
            if msg["type"] != "message":
                continue

            payload = msg["data"]
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")

            await handler(json.loads(payload))
