"""Reclaiming a dead worker's stream entries, and only a dead worker's
(P3-6 §7.3, §14 "AI-service (Redis)"). Real Redis (TEST_REDIS_URL)."""

import asyncio
import json
import os
from uuid import uuid4

import pytest
import redis.asyncio as redis_asyncio

from app.services.tasks.reclaimer import StreamReclaimer
from app.services.tasks.task_records import (
    ReclaimResult,
    RedisTaskRecordStore,
    TaskState,
    WriteResult,
)
from tests.unit.test_task_records import record

REDIS_URL = os.getenv("TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL not set")

GROUP = "ai_worker_group"
LEASE = 0.15  # candidates after 0.3 s idle


@pytest.fixture
async def env():
    client = redis_asyncio.Redis.from_url(REDIS_URL, decode_responses=True)
    store = RedisTaskRecordStore.connect(REDIS_URL, timeout=2.0)
    stream = f"{{ai-tasks}}:test:{uuid4().hex}"
    await client.xgroup_create(stream, GROUP, id="0", mkstream=True)
    reclaimer = StreamReclaimer(client, store, group=GROUP, lease_seconds=LEASE, stream=stream)
    keys = [stream]
    try:
        yield client, store, stream, reclaimer, keys
    finally:
        await client.delete(*keys)
        await client.close()
        await store.close()


async def deliver(client, stream, keys, consumer, task_type="content.ingestion"):
    """Publish a task and read it as ``consumer``: the entry is then pending."""
    task_id = f"task_{uuid4().hex}"
    keys.append(RedisTaskRecordStore.key(task_id))
    await client.xadd(stream, {"data": json.dumps({"task_id": task_id, "task_type": task_type})})
    read = await client.xreadgroup(GROUP, consumer, {stream: ">"}, count=1)
    return task_id, read[0][1][0][0]


async def pending(client, stream):
    return [p["message_id"] for p in await client.xpending_range(stream, GROUP, "-", "+", 100)]


async def test_a_dead_consumers_entry_is_lost_and_acknowledged_without_running(env):
    client, store, stream, reclaimer, keys = env
    task_id, entry = await deliver(client, stream, keys, "dead")
    await store.create(record(task_id))
    await store.start(task_id, "dead", LEASE)  # then the worker dies: no renewals

    assert await reclaimer.run_once() == {}  # not idle long enough yet
    await asyncio.sleep(LEASE * 2.5)
    assert await reclaimer.run_once() == {ReclaimResult.LOST: 1}

    assert await pending(client, stream) == []
    stored = await store.get(task_id)
    assert (stored.state, stored.lost_entry) == (TaskState.LOST, entry)
    # A worker that later receives the task again refuses to run it.
    assert await store.start(task_id, "other", 60) == WriteResult.FENCED


async def test_a_healthy_handler_is_never_reclaimed(env):
    """Renewing across five lease windows while its entry stays pending well
    past the idle threshold, with reclaim passes throughout."""
    client, store, stream, reclaimer, keys = env
    task_id, entry = await deliver(client, stream, keys, "healthy")
    await store.create(record(task_id))
    await store.start(task_id, "healthy", LEASE)

    outcomes = []
    for _ in range(15):  # five lease windows, renewing every third of one
        await asyncio.sleep(LEASE / 3)
        assert await store.renew(task_id, "healthy", LEASE) == WriteResult.OK
        outcomes.append(await reclaimer.run_once())

    assert {ReclaimResult.LOST, ReclaimResult.ACKED} & {k for o in outcomes for k in o} == set()
    assert {ReclaimResult.LIVE: 1} in outcomes  # it was a candidate, and judged live
    assert await pending(client, stream) == [entry]
    live = await store.get(task_id)
    assert (live.state, live.worker_id) == (TaskState.RUNNING, "healthy")
    assert await store.record_event(task_id, "healthy", state=TaskState.SUCCEEDED) == "ok"


async def test_a_task_that_never_started_is_reclaimed_after_the_threshold(env):
    client, store, stream, reclaimer, keys = env
    task_id, _ = await deliver(client, stream, keys, "died-before-start")
    await store.create(record(task_id))
    await asyncio.sleep(LEASE * 2.5)
    assert await reclaimer.run_once() == {ReclaimResult.LOST: 1}
    assert await pending(client, stream) == []


async def test_a_reclaimer_that_crashes_before_the_script_is_completed_by_the_next_pass(env):
    client, store, stream, reclaimer, keys = env
    task_id, _ = await deliver(client, stream, keys, "dead")
    await store.create(record(task_id))
    await store.start(task_id, "dead", LEASE)
    await asyncio.sleep(LEASE * 2.5)

    class Crash(Exception):
        pass

    real = store.reclaim

    async def crash(*args, **kwargs):
        raise Crash  # after listing the candidates, before the script

    store.reclaim = crash
    with pytest.raises(Crash):
        await reclaimer.run_once()
    assert len(await pending(client, stream)) == 1
    store.reclaim = real

    assert await reclaimer.run_once() == {ReclaimResult.LOST: 1}  # the restart
    assert await reclaimer.run_once() == {}  # after the script: nothing left
    assert await pending(client, stream) == []


async def test_an_entry_left_pending_by_an_interrupted_reclaim_is_acknowledged(env):
    """The record is already lost for the entry, or has expired, but the
    entry is still pending: the next pass only acknowledges it."""
    client, store, stream, reclaimer, keys = env
    lost_id, lost_entry = await deliver(client, stream, keys, "dead")
    await store.create(record(lost_id, state=TaskState.LOST))
    await client.hset(RedisTaskRecordStore.key(lost_id), "lost_entry", lost_entry)
    await deliver(client, stream, keys, "dead")  # its record expired: none exists
    await asyncio.sleep(LEASE * 2.5)

    assert await reclaimer.run_once() == {ReclaimResult.ACKED: 1, ReclaimResult.GONE: 1}
    assert await pending(client, stream) == []


async def test_entries_of_other_task_types_are_left_alone(env):
    client, store, stream, reclaimer, keys = env
    _, entry = await deliver(client, stream, keys, "dead", task_type="agent.execution")
    await asyncio.sleep(LEASE * 2.5)
    assert await reclaimer.run_once() == {"skipped": 1}
    assert await pending(client, stream) == [entry]
