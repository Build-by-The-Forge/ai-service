"""Ingestion task records for status reconciliation (M3-09)."""

import os
from uuid import uuid4

import pytest
import redis.asyncio as redis_asyncio

from app.services.tasks.task_records import (
    MAX_RECORDED_EVENTS,
    TASK_RECORD_TTL_SECONDS,
    InMemoryTaskRecordStore,
    RedisTaskRecordStore,
    TaskRecord,
    TaskRecordUnavailableError,
    TaskState,
)

REDIS_URL = os.getenv("TEST_REDIS_URL")
real_redis = pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL not set")
TENANT = "6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c"


def record(task_id="task_1", **kwargs) -> TaskRecord:
    fields = dict(
        external_task_id=task_id, tenant_id=TENANT, job_id="job_1", run_number=1, attempt_count=2
    )
    fields.update(kwargs)
    return TaskRecord(**fields)


# --- the store ---------------------------------------------------------------------


async def test_in_memory_create_keeps_the_first_record():
    store = InMemoryTaskRecordStore()
    await store.create(record(job_id="first"))
    await store.create(record(job_id="second"))
    assert (await store.get("task_1")).job_id == "first"


async def test_updates_append_events_and_set_state_up_to_a_bound():
    store = InMemoryTaskRecordStore()
    await store.create(record())
    for i in range(MAX_RECORDED_EVENTS + 5):
        await store.update("task_1", event={"n": i})
    await store.update("task_1", state=TaskState.FAILED)
    stored = await store.get("task_1")
    assert len(stored.events) == MAX_RECORDED_EVENTS
    assert stored.state == TaskState.FAILED


async def test_updating_an_unknown_task_does_nothing():
    store = InMemoryTaskRecordStore()
    await store.update("missing", state=TaskState.RUNNING)
    assert await store.get("missing") is None


@real_redis
async def test_redis_store_round_trip_with_expiry():
    store = RedisTaskRecordStore.connect(REDIS_URL, timeout=2.0)
    task_id = f"task_{uuid4().hex}"
    try:
        await store.create(record(task_id))
        await store.create(record(task_id, job_id="other"))  # NX: first wins
        await store.update(task_id, state=TaskState.RUNNING, event={"event_type": "job_failed"})
        stored = await store.get(task_id)
        assert (stored.job_id, stored.state, stored.events) == (
            "job_1",
            TaskState.RUNNING,
            [{"event_type": "job_failed"}],
        )
        client = redis_asyncio.Redis.from_url(REDIS_URL)
        ttl = await client.ttl(RedisTaskRecordStore.key(task_id))
        await client.close()
        assert 0 < ttl <= TASK_RECORD_TTL_SECONDS
    finally:
        await store.close()


async def test_unreachable_redis_is_reported():
    store = RedisTaskRecordStore.connect("redis://127.0.0.1:1/0", timeout=0.3)
    try:
        with pytest.raises(TaskRecordUnavailableError):
            await store.get("task_x")
        with pytest.raises(TaskRecordUnavailableError):
            await store.create(record())
    finally:
        await store.close()
