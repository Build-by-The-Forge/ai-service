import abc
import asyncio
import json
from typing import Any

from app.core.execution_context import AITask
from app.core.redis import get_redis_client


class TaskTransport(abc.ABC):
    """
    Abstract Transport Port decoupling task publishing, consumption, acknowledgments,
    and DLQ routing from underlying transport engines (Redis Streams, InMemory, Kafka, SQS).
    """

    @abc.abstractmethod
    async def publish(self, task: AITask, stream_name: str = "stream:ai_tasks") -> str:
        """Publishes an AITask envelope to the designated stream. Returns the transport message ID."""

    @abc.abstractmethod
    async def consume(
        self,
        group: str = "ai_worker_group",
        consumer: str = "worker_1",
        stream_name: str = "stream:ai_tasks",
        count: int = 1,
        block_ms: int = 2000,
    ) -> list[tuple[str, AITask]]:
        """
        Consumes tasks from the stream.
        Returns a list of (transport_msg_id, AITask).
        """

    @abc.abstractmethod
    async def ack(
        self,
        stream_msg_id: str,
        stream_name: str = "stream:ai_tasks",
        group: str = "ai_worker_group",
    ) -> None:
        """Acknowledges successful task processing."""

    @abc.abstractmethod
    async def send_to_dlq(
        self,
        task: AITask,
        error_reason: str,
        dlq_stream_name: str = "stream:ai_tasks:dlq",
    ) -> str:
        """Transfers a permanently failed or max-retried task to the Dead-Letter Queue."""


class InMemoryTaskTransport(TaskTransport):
    """
    In-memory async transport for local testing and isolated execution.
    """

    def __init__(self):
        self._queue: asyncio.Queue[tuple[str, AITask]] = asyncio.Queue()
        self._dlq: list[dict[str, Any]] = []
        self._counter = 0
        self._acked: set[str] = set()

    async def publish(self, task: AITask, stream_name: str = "stream:ai_tasks") -> str:
        self._counter += 1
        msg_id = f"mem_msg_{self._counter}"
        await self._queue.put((msg_id, task))
        return msg_id

    async def consume(
        self,
        group: str = "ai_worker_group",
        consumer: str = "worker_1",
        stream_name: str = "stream:ai_tasks",
        count: int = 1,
        block_ms: int = 2000,
    ) -> list[tuple[str, AITask]]:
        items: list[tuple[str, AITask]] = []
        try:
            timeout_sec = max(block_ms / 1000.0, 0.01)
            msg_id, task = await asyncio.wait_for(self._queue.get(), timeout=timeout_sec)
            items.append((msg_id, task))
        except TimeoutError:
            pass
        return items

    async def ack(
        self,
        stream_msg_id: str,
        stream_name: str = "stream:ai_tasks",
        group: str = "ai_worker_group",
    ) -> None:
        self._acked.add(stream_msg_id)

    async def send_to_dlq(
        self,
        task: AITask,
        error_reason: str,
        dlq_stream_name: str = "stream:ai_tasks:dlq",
    ) -> str:
        self._counter += 1
        dlq_id = f"dlq_msg_{self._counter}"
        self._dlq.append(
            {
                "dlq_id": dlq_id,
                "task": task.to_dict(),
                "error_reason": error_reason,
            }
        )
        return dlq_id

    @property
    def dlq_items(self) -> list[dict[str, Any]]:
        return self._dlq


class RedisStreamTransport(TaskTransport):
    """
    Production-grade Redis Streams transport using consumer groups (XADD, XREADGROUP, XACK).
    """

    def __init__(self, fallback: TaskTransport | None = None):
        self._fallback = fallback or InMemoryTaskTransport()
        self._groups_created: set[str] = set()

    async def _ensure_group(self, stream_name: str, group: str) -> None:
        group_key = f"{stream_name}:{group}"
        if group_key in self._groups_created:
            return
        try:
            client = get_redis_client()
            await client.xgroup_create(stream_name, group, id="0", mkstream=True)
            self._groups_created.add(group_key)
        except Exception as e:
            # Group may already exist (BUSYGROUP)
            if "BUSYGROUP" in str(e):
                self._groups_created.add(group_key)
            else:
                pass

    async def publish(self, task: AITask, stream_name: str = "stream:ai_tasks") -> str:
        try:
            client = get_redis_client()
            task_json = json.dumps(task.to_dict())
            msg_id = await client.xadd(stream_name, {"data": task_json})
            return str(msg_id)
        except Exception:
            return await self._fallback.publish(task, stream_name)

    async def consume(
        self,
        group: str = "ai_worker_group",
        consumer: str = "worker_1",
        stream_name: str = "stream:ai_tasks",
        count: int = 1,
        block_ms: int = 2000,
    ) -> list[tuple[str, AITask]]:
        try:
            await self._ensure_group(stream_name, group)
            client = get_redis_client()
            # Read new messages with '>'
            raw_entries = await client.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams={stream_name: ">"},
                count=count,
                block=block_ms,
            )
            results: list[tuple[str, AITask]] = []
            if raw_entries:
                for stream, messages in raw_entries:
                    for msg_id, data in messages:
                        payload_str = data.get("data")
                        if payload_str:
                            task_dict = json.loads(payload_str)
                            results.append((str(msg_id), AITask.from_dict(task_dict)))
            return results
        except Exception:
            return await self._fallback.consume(group, consumer, stream_name, count, block_ms)

    async def ack(
        self,
        stream_msg_id: str,
        stream_name: str = "stream:ai_tasks",
        group: str = "ai_worker_group",
    ) -> None:
        try:
            client = get_redis_client()
            await client.xack(stream_name, group, stream_msg_id)
        except Exception:
            await self._fallback.ack(stream_msg_id, stream_name, group)

    async def send_to_dlq(
        self,
        task: AITask,
        error_reason: str,
        dlq_stream_name: str = "stream:ai_tasks:dlq",
    ) -> str:
        try:
            client = get_redis_client()
            envelope = {
                "task": json.dumps(task.to_dict()),
                "error_reason": error_reason,
            }
            msg_id = await client.xadd(dlq_stream_name, envelope)
            return str(msg_id)
        except Exception:
            return await self._fallback.send_to_dlq(task, error_reason, dlq_stream_name)


# Global Transport Provider
_default_transport: TaskTransport | None = None


def get_task_transport() -> TaskTransport:
    global _default_transport
    if _default_transport is None:
        _default_transport = RedisStreamTransport()
    return _default_transport


def set_task_transport(transport: TaskTransport) -> None:
    global _default_transport
    _default_transport = transport
