"""What AI-service knows about each ingestion task, for status reconciliation
(M3-09, design B.3 "Status reconciliation").

One record per external task id: the tenant it belongs to (from the verified
dispatch context), the backend attempt identity, the task's state, and the
exact backend callback bodies the worker produced, in order, each with its
event_id. The backend's reconcile operation replays those bodies through the
same callback rules it applies to delivered callbacks, so a callback that was
never delivered can still be applied, and one that was is a duplicate.

Records live in Redis only: an in-process fallback would give replicas
different answers. They expire after TASK_RECORD_TTL_SECONDS.
"""

from __future__ import annotations

import abc
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import redis.asyncio as redis_asyncio
from redis.exceptions import RedisError

TASK_RECORD_TTL_SECONDS = 14 * 24 * 3600
MAX_RECORDED_EVENTS = 20


class TaskState:
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TaskRecordUnavailableError(Exception):
    """The record store could not be reached."""


@dataclass
class TaskRecord:
    external_task_id: str
    tenant_id: str
    job_id: str
    run_number: int
    attempt_count: int
    state: str = TaskState.ACCEPTED
    events: list[dict[str, Any]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


class TaskRecordStore(abc.ABC):
    @abc.abstractmethod
    async def create(self, record: TaskRecord) -> None:
        """Store a new record; an existing record for the task is kept."""

    @abc.abstractmethod
    async def get(self, external_task_id: str) -> TaskRecord | None: ...

    @abc.abstractmethod
    async def save(self, record: TaskRecord) -> None: ...

    async def update(
        self,
        external_task_id: str,
        *,
        state: str | None = None,
        event: dict[str, Any] | None = None,
    ) -> None:
        record = await self.get(external_task_id)
        if record is None:
            return
        if state is not None:
            record.state = state
        if event is not None and len(record.events) < MAX_RECORDED_EVENTS:
            record.events.append(event)
        record.updated_at = time.time()
        await self.save(record)


class InMemoryTaskRecordStore(TaskRecordStore):
    """For tests only."""

    def __init__(self) -> None:
        self.records: dict[str, TaskRecord] = {}

    async def create(self, record: TaskRecord) -> None:
        self.records.setdefault(record.external_task_id, record)

    async def get(self, external_task_id: str) -> TaskRecord | None:
        return self.records.get(external_task_id)

    async def save(self, record: TaskRecord) -> None:
        self.records[record.external_task_id] = record


class RedisTaskRecordStore(TaskRecordStore):
    def __init__(self, client: redis_asyncio.Redis) -> None:
        self._client = client

    @classmethod
    def connect(cls, url: str, *, timeout: float = 1.0) -> RedisTaskRecordStore:
        return cls(
            redis_asyncio.Redis.from_url(
                url, decode_responses=True, socket_timeout=timeout, socket_connect_timeout=timeout
            )
        )

    @staticmethod
    def key(external_task_id: str) -> str:
        return f"ingest:task:{external_task_id}"

    async def create(self, record: TaskRecord) -> None:
        try:
            await self._client.set(
                self.key(record.external_task_id),
                json.dumps(asdict(record)),
                nx=True,
                ex=TASK_RECORD_TTL_SECONDS,
            )
        except (RedisError, OSError) as exc:
            raise TaskRecordUnavailableError(type(exc).__name__) from exc

    async def get(self, external_task_id: str) -> TaskRecord | None:
        try:
            raw = await self._client.get(self.key(external_task_id))
        except (RedisError, OSError) as exc:
            raise TaskRecordUnavailableError(type(exc).__name__) from exc
        return TaskRecord(**json.loads(raw)) if raw else None

    async def save(self, record: TaskRecord) -> None:
        # One worker processes a task at a time (stream consumer group), so a
        # read-modify-write per record does not race in practice.
        try:
            await self._client.set(
                self.key(record.external_task_id),
                json.dumps(asdict(record)),
                ex=TASK_RECORD_TTL_SECONDS,
            )
        except (RedisError, OSError) as exc:
            raise TaskRecordUnavailableError(type(exc).__name__) from exc

    async def close(self) -> None:
        await self._client.close()  # redis-py 5.0.0 has no aclose()


_default_store: TaskRecordStore | None = None


def get_task_record_store() -> TaskRecordStore:
    global _default_store
    if _default_store is None:
        from app.core.config import settings

        _default_store = RedisTaskRecordStore.connect(settings.REDIS_URL)
    return _default_store


def set_task_record_store(store: TaskRecordStore | None) -> None:
    global _default_store
    _default_store = store
