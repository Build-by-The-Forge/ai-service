"""Ingestion task records for status reconciliation (M3-09) and task
liveness (P3-6 §7): the lease, its conditional writes, and reclaim.

The rules are tested against both stores: the in-memory one (the reference
for unit tests) and Redis (the scripts production runs, TEST_REDIS_URL).
"""

import asyncio
import json
import os
import time
from uuid import uuid4

import pytest
import redis.asyncio as redis_asyncio

from app.services.tasks.task_records import (
    MAX_RECORDED_EVENTS,
    TASK_RECORD_TTL_SECONDS,
    InMemoryTaskRecordStore,
    ReclaimResult,
    RedisTaskRecordStore,
    TaskRecord,
    TaskRecordUnavailableError,
    TaskState,
    WriteResult,
    reported_state,
)

REDIS_URL = os.getenv("TEST_REDIS_URL")
real_redis = pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL not set")
TENANT = "6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c"
GROUP = "ai_worker_group"
SHORT = 0.05  # a lease that lapses within the test


def record(task_id="task_1", **kwargs) -> TaskRecord:
    fields = dict(
        external_task_id=task_id, tenant_id=TENANT, job_id="job_1", run_number=1, attempt_count=2
    )
    fields.update(kwargs)
    return TaskRecord(**fields)


class Harness:
    """A store plus a stream with one pending entry per task, so reclaim's
    acknowledgement can be observed."""

    def __init__(self, store, client=None):
        self.store, self.client = store, client
        self.stream = f"{{ai-tasks}}:test:{uuid4().hex}"
        self.task_id = f"task_{uuid4().hex}"

    async def pending_entry(self, consumer: str) -> str:
        if self.client is None:
            return f"entry-{uuid4().hex}"
        await self.client.xgroup_create(self.stream, GROUP, id="0", mkstream=True)
        await self.client.xadd(self.stream, {"data": "{}"})
        read = await self.client.xreadgroup(GROUP, consumer, {self.stream: ">"}, count=1)
        return read[0][1][0][0]

    async def pending(self, entry: str) -> bool:
        if self.client is None:
            return entry not in self.store.acked
        rows = await self.client.xpending_range(self.stream, GROUP, min=entry, max=entry, count=1)
        return bool(rows)

    async def reclaim(self, entry: str, consumer: str, idle_ms: int = 0, min_idle_ms: int = 1):
        return await self.store.reclaim(
            self.task_id,
            entry_id=entry,
            consumer=consumer,
            idle_ms=idle_ms,
            min_idle_ms=min_idle_ms,
            stream=self.stream,
            group=GROUP,
        )


@pytest.fixture(params=["memory", pytest.param("redis", marks=real_redis)])
async def h(request):
    if request.param == "memory":
        yield Harness(InMemoryTaskRecordStore())
        return
    store = RedisTaskRecordStore.connect(REDIS_URL, timeout=2.0)
    client = redis_asyncio.Redis.from_url(REDIS_URL, decode_responses=True)
    harness = Harness(store, client)
    try:
        yield harness
    finally:
        await client.delete(harness.stream, RedisTaskRecordStore.key(harness.task_id))
        await client.close()
        await store.close()


# --- records and events ------------------------------------------------------------


async def test_create_keeps_the_first_record(h):
    await h.store.create(record(h.task_id, job_id="first"))
    await h.store.create(record(h.task_id, job_id="second"))
    stored = await h.store.get(h.task_id)
    assert (stored.job_id, stored.state, stored.events) == ("first", TaskState.ACCEPTED, [])


async def test_unknown_tasks_are_missing(h):
    assert await h.store.get(h.task_id) is None
    assert await h.store.start(h.task_id, "w1", 60) == WriteResult.MISSING
    assert await h.store.renew(h.task_id, "w1", 60) == WriteResult.MISSING
    assert await h.store.record_event(h.task_id, "w1", event={"n": 1}) == WriteResult.MISSING
    assert await h.store.release(h.task_id, "w1") == WriteResult.MISSING
    assert await h.store.get(h.task_id) is None  # no write creates a record


async def test_events_are_appended_up_to_a_bound(h):
    await h.store.create(record(h.task_id))
    assert await h.store.start(h.task_id, "w1", 60) == WriteResult.OK
    for i in range(MAX_RECORDED_EVENTS + 5):
        await h.store.record_event(h.task_id, "w1", event={"n": i}, lease_seconds=60)
    await h.store.record_event(h.task_id, "w1", state=TaskState.FAILED)
    stored = await h.store.get(h.task_id)
    assert [e["n"] for e in stored.events] == list(range(MAX_RECORDED_EVENTS))
    assert stored.state == TaskState.FAILED


async def test_nothing_is_recorded_for_a_task_nobody_has_started(h):
    await h.store.create(record(h.task_id))
    assert await h.store.record_event(h.task_id, "w1", event={"n": 1}) == WriteResult.FENCED
    assert await h.store.record_event(h.task_id, "w1", state=TaskState.FAILED) == (
        WriteResult.FENCED
    )
    stored = await h.store.get(h.task_id)
    assert (stored.state, stored.events) == (TaskState.ACCEPTED, [])


# --- the lease ---------------------------------------------------------------------


async def test_start_takes_the_lease_once(h):
    await h.store.create(record(h.task_id))
    before = time.time()
    assert await h.store.start(h.task_id, "w1", 60) == WriteResult.OK
    assert await h.store.start(h.task_id, "w2", 60) == WriteResult.FENCED
    stored = await h.store.get(h.task_id)
    assert (stored.state, stored.worker_id) == (TaskState.RUNNING, "w1")
    assert before + 59 < stored.lease_until < before + 62
    assert reported_state(stored) == TaskState.RUNNING


async def test_renewal_extends_only_the_owners_lease(h):
    await h.store.create(record(h.task_id))
    await h.store.start(h.task_id, "w1", SHORT)
    await asyncio.sleep(SHORT * 2)
    assert reported_state(await h.store.get(h.task_id)) == TaskState.STALLED
    assert await h.store.renew(h.task_id, "w2", 60) == WriteResult.FENCED
    assert await h.store.renew(h.task_id, "w1", 60) == WriteResult.OK
    assert reported_state(await h.store.get(h.task_id)) == TaskState.RUNNING


async def test_a_terminal_state_clears_the_lease_and_fences_later_writes(h):
    await h.store.create(record(h.task_id))
    await h.store.start(h.task_id, "w1", 60)
    await h.store.record_event(
        h.task_id, "w1", event={"event_type": "job_completed"}, state=TaskState.SUCCEEDED
    )
    stored = await h.store.get(h.task_id)
    assert (stored.state, stored.lease_until) == (TaskState.SUCCEEDED, None)
    assert reported_state(stored) == TaskState.SUCCEEDED
    assert await h.store.renew(h.task_id, "w1", 60) == WriteResult.FENCED
    assert await h.store.record_event(h.task_id, "w1", event={"n": 2}) == WriteResult.FENCED
    assert len((await h.store.get(h.task_id)).events) == 1


async def test_another_workers_writes_are_refused(h):
    await h.store.create(record(h.task_id))
    await h.store.start(h.task_id, "w1", 60)
    assert await h.store.record_event(h.task_id, "w2", event={"n": 1}) == WriteResult.FENCED
    assert await h.store.release(h.task_id, "w2") == WriteResult.FENCED
    assert (await h.store.get(h.task_id)).events == []


async def test_release_hands_the_task_to_the_next_delivery(h):
    await h.store.create(record(h.task_id))
    await h.store.start(h.task_id, "w1", 60)
    assert await h.store.release(h.task_id, "w1") == WriteResult.OK
    stored = await h.store.get(h.task_id)
    assert (stored.state, stored.worker_id, stored.lease_until) == (
        TaskState.ACCEPTED,
        None,
        None,
    )
    assert await h.store.start(h.task_id, "w2", 60) == WriteResult.OK
    assert await h.store.renew(h.task_id, "w1", 60) == WriteResult.FENCED


# --- reclaim -----------------------------------------------------------------------


async def test_a_live_lease_is_never_reclaimed(h):
    await h.store.create(record(h.task_id))
    entry = await h.pending_entry("w1")
    await h.store.start(h.task_id, "w1", 60)
    assert await h.reclaim(entry, "w1", idle_ms=10**9) == ReclaimResult.LIVE
    assert await h.pending(entry)
    assert (await h.store.get(h.task_id)).state == TaskState.RUNNING


async def test_an_expired_lease_of_another_consumer_is_untouched(h):
    await h.store.create(record(h.task_id))
    entry = await h.pending_entry("w1")
    await h.store.start(h.task_id, "w2", SHORT)
    await asyncio.sleep(SHORT * 2)
    assert await h.reclaim(entry, "w1") == ReclaimResult.LIVE
    assert await h.pending(entry)


async def test_a_dead_consumers_task_is_lost_and_acknowledged_together(h):
    await h.store.create(record(h.task_id))
    entry = await h.pending_entry("w1")
    await h.store.start(h.task_id, "w1", SHORT)
    await asyncio.sleep(SHORT * 2)

    assert await h.reclaim(entry, "w1") == ReclaimResult.LOST
    stored = await h.store.get(h.task_id)
    assert (stored.state, stored.lost_entry) == (TaskState.LOST, entry)
    assert not await h.pending(entry)
    # The fenced worker can no longer renew, record or release.
    assert await h.store.renew(h.task_id, "w1", 60) == WriteResult.FENCED
    assert await h.store.record_event(h.task_id, "w1", event={"n": 1}) == WriteResult.FENCED
    assert await h.store.release(h.task_id, "w1") == WriteResult.FENCED
    # Idempotent: a repeat only acknowledges.
    assert await h.reclaim(entry, "w1") == ReclaimResult.ACKED


async def test_an_entry_left_pending_after_lost_is_acknowledged(h):
    await h.store.create(record(h.task_id, state=TaskState.LOST, lost_entry="placeholder"))
    entry = await h.pending_entry("w1")
    if h.client is not None:
        await h.client.hset(RedisTaskRecordStore.key(h.task_id), "lost_entry", entry)
    else:
        h.store.records[h.task_id].lost_entry = entry
    assert await h.reclaim(entry, "w1") == ReclaimResult.ACKED
    assert not await h.pending(entry)


async def test_an_entry_whose_record_expired_is_acknowledged(h):
    entry = await h.pending_entry("w1")
    assert await h.reclaim(entry, "w1") == ReclaimResult.GONE
    assert not await h.pending(entry)


async def test_a_task_never_started_is_lost_only_after_the_idle_threshold(h):
    await h.store.create(record(h.task_id))
    entry = await h.pending_entry("w1")
    assert await h.reclaim(entry, "w1", idle_ms=999, min_idle_ms=1000) == ReclaimResult.LIVE
    assert await h.reclaim(entry, "w1", idle_ms=1000, min_idle_ms=1000) == ReclaimResult.LOST
    assert await h.store.start(h.task_id, "w1", 60) == WriteResult.FENCED


# --- Redis only --------------------------------------------------------------------


@real_redis
async def test_recorded_events_keep_their_exact_values():
    store = RedisTaskRecordStore.connect(REDIS_URL, timeout=2.0)
    task_id = f"task_{uuid4().hex}"
    event = {"seconds": 0.1234567890123456, "size": 12345678901234567, "text": 'é ≤ "q"'}
    try:
        await store.create(record(task_id))
        await store.start(task_id, "w1", 60)
        await store.record_event(task_id, "w1", event=event, lease_seconds=60)
        await store.record_event(task_id, "w1", event={"n": 2}, lease_seconds=60)
        assert (await store.get(task_id)).events == [event, {"n": 2}]
        assert json.dumps((await store.get(task_id)).events[0]) == json.dumps(event)
    finally:
        await store.close()


@real_redis
async def test_renewal_resets_the_record_ttl():
    store = RedisTaskRecordStore.connect(REDIS_URL, timeout=2.0)
    client = redis_asyncio.Redis.from_url(REDIS_URL)
    task_id = f"task_{uuid4().hex}"
    key = RedisTaskRecordStore.key(task_id)
    try:
        await store.create(record(task_id))
        assert 0 < await client.ttl(key) <= TASK_RECORD_TTL_SECONDS
        await store.start(task_id, "w1", 60)
        await client.expire(key, 30)
        assert await store.renew(task_id, "w1", 60) == WriteResult.OK
        assert await client.ttl(key) > TASK_RECORD_TTL_SECONDS - 5
    finally:
        await client.close()
        await store.close()


@real_redis
async def test_record_expiry_under_a_running_task_fences_it():
    store = RedisTaskRecordStore.connect(REDIS_URL, timeout=2.0)
    client = redis_asyncio.Redis.from_url(REDIS_URL)
    task_id = f"task_{uuid4().hex}"
    try:
        await store.create(record(task_id))
        await store.start(task_id, "w1", 60)
        await client.delete(RedisTaskRecordStore.key(task_id))
        assert await store.renew(task_id, "w1", 60) == WriteResult.MISSING
        assert await store.record_event(task_id, "w1", event={"n": 1}) == WriteResult.MISSING
        assert await store.get(task_id) is None
    finally:
        await client.close()
        await store.close()


@real_redis
async def test_concurrent_renewal_and_reclaim_have_exactly_one_outcome():
    """P3-6 §14 "the race": either the lease is extended and nothing is
    reclaimed, or the task is lost and the renewal refused. Never both."""
    store = RedisTaskRecordStore.connect(REDIS_URL, timeout=2.0)
    client = redis_asyncio.Redis.from_url(REDIS_URL, decode_responses=True)
    outcomes = set()
    try:
        for _ in range(50):
            h = Harness(store, client)
            await store.create(record(h.task_id))
            entry = await h.pending_entry("w1")
            await store.start(h.task_id, "w1", SHORT / 5)
            await asyncio.sleep(SHORT / 2)  # the lease has lapsed; both race
            renewed, reclaimed = await asyncio.gather(
                store.renew(h.task_id, "w1", 60), h.reclaim(entry, "w1")
            )
            state = (await store.get(h.task_id)).state
            pending = await h.pending(entry)
            if renewed == WriteResult.OK:
                assert (reclaimed, state, pending) == (ReclaimResult.LIVE, TaskState.RUNNING, True)
            else:
                assert (renewed, reclaimed, state, pending) == (
                    WriteResult.FENCED,
                    ReclaimResult.LOST,
                    TaskState.LOST,
                    False,
                )
            outcomes.add(renewed)
            await client.delete(h.stream, RedisTaskRecordStore.key(h.task_id))
    finally:
        await client.close()
        await store.close()
    assert outcomes <= {WriteResult.OK, WriteResult.FENCED}


async def test_unreachable_redis_is_reported():
    store = RedisTaskRecordStore.connect("redis://127.0.0.1:1/0", timeout=0.3)
    try:
        with pytest.raises(TaskRecordUnavailableError):
            await store.get("task_x")
        with pytest.raises(TaskRecordUnavailableError):
            await store.create(record())
        with pytest.raises(TaskRecordUnavailableError):
            await store.renew("task_x", "w1", 60)
    finally:
        await store.close()
