"""An ingestion task runs under a renewed lease, and a worker that loses its
task stops (P3-6 §7.1).

The worker takes the lease when it starts the task, renews it while the
handler runs, and clears it at the end. When a write is refused (the record
was reclaimed or lost), the task is fenced: the handler is cancelled, and no
further event is recorded or delivered.
"""

import asyncio
import time

from app.core.callbacks.dispatcher import InMemoryCallbackDispatcher
from app.core.callbacks.events import CallbackEventType
from app.core.execution_context import TaskStatus
from app.services.tasks.handlers import TaskHandler, TaskHandlerRegistry
from app.services.tasks.ingestion_outcomes import TransientIngestionError
from app.services.tasks.task_records import InMemoryTaskRecordStore, TaskState, WriteResult
from app.services.tasks.transport import InMemoryTaskTransport
from app.services.tasks.worker import TaskWorker
from tests.unit.test_task_records import record
from tests.unit.test_worker_task_records import ingestion_task

LEASE = 0.3  # renewed every 0.1 s


class Gated(TaskHandler):
    """Reports ``extracted``, then waits for the test before going on."""

    def __init__(self, hold: float = 0):
        self.go = asyncio.Event()
        self.extracted = asyncio.Event()
        self.hold = hold
        self.finished = False

    async def handle(self, ctx, payload):
        await ctx.stage_reporter.report("extracted", {"characters": 9, "pages": 1})
        self.extracted.set()
        await asyncio.sleep(self.hold)
        await self.go.wait()
        await ctx.stage_reporter.report("chunked", {"chunks": 2})
        await ctx.stage_reporter.report("indexed", {"indexed_chunks": 2})
        self.finished = True
        return {}


def worker(handler, store, worker_id="w1"):
    registry = TaskHandlerRegistry()
    registry.register("content.ingestion", handler)
    dispatcher = InMemoryCallbackDispatcher()
    transport = InMemoryTaskTransport()
    return (
        TaskWorker(
            worker_id=worker_id,
            transport=transport,
            registry=registry,
            callback_dispatcher=dispatcher,
            task_records=store,
            lease_seconds=LEASE,
        ),
        dispatcher,
        transport,
    )


def kinds(dispatcher):
    return [(e.event_type, e.stage) for e in dispatcher.delivered_events]


async def test_the_lease_is_renewed_across_many_windows_then_cleared():
    store = InMemoryTaskRecordStore()
    await store.create(record())
    handler = Gated(hold=LEASE * 6)
    handler.go.set()
    w, dispatcher, _ = worker(handler, store)

    run = asyncio.create_task(w.process_single_task("m1", ingestion_task()))
    await handler.extracted.wait()
    for _ in range(5):  # five lease windows: the lease never lapses
        await asyncio.sleep(LEASE)
        live = await store.get("task_1")
        assert (live.state, live.worker_id) == (TaskState.RUNNING, "w1")
        assert live.lease_until > time.time()

    assert await run == TaskStatus.SUCCEEDED
    done = await store.get("task_1")
    assert (done.state, done.lease_until) == (TaskState.SUCCEEDED, None)


async def test_a_lost_record_stops_the_handler_and_everything_after_it():
    """§14 "record expiry during an active handler"."""
    store = InMemoryTaskRecordStore()
    await store.create(record())
    handler = Gated()
    w, dispatcher, transport = worker(handler, store)

    run = asyncio.create_task(w.process_single_task("m1", ingestion_task()))
    await handler.extracted.wait()
    del store.records["task_1"]  # expired or lost
    status = await asyncio.wait_for(run, timeout=LEASE * 3)

    assert status == TaskStatus.CANCELLED
    assert not handler.finished
    assert kinds(dispatcher) == [
        (CallbackEventType.STARTED, None),
        (CallbackEventType.STAGE_COMPLETED, "extracted"),
    ]
    assert "m1" in transport._acked


async def test_a_reclaimed_task_is_fenced_at_its_next_report():
    store = InMemoryTaskRecordStore()
    await store.create(record())
    handler = Gated()
    w, dispatcher, _ = worker(handler, store, worker_id="w1")
    w.lease_seconds = 60  # no renewal before the report: the report is the check

    run = asyncio.create_task(w.process_single_task("m1", ingestion_task()))
    await handler.extracted.wait()
    store.records["task_1"].state = TaskState.LOST  # reclaimed meanwhile
    handler.go.set()

    assert await run == TaskStatus.CANCELLED
    assert not handler.finished
    stored = await store.get("task_1")
    assert stored.state == TaskState.LOST
    assert [e.get("stage") for e in stored.events] == ["extracted"]
    assert (CallbackEventType.STAGE_COMPLETED, "chunked") not in kinds(dispatcher)
    assert CallbackEventType.COMPLETED not in [k for k, _ in kinds(dispatcher)]


async def test_a_task_held_elsewhere_or_finished_is_not_run():
    for state in (TaskState.RUNNING, TaskState.LOST, TaskState.SUCCEEDED):
        store = InMemoryTaskRecordStore()
        await store.create(record(state=state, worker_id="w0"))
        handler = Gated()
        handler.go.set()
        w, dispatcher, transport = worker(handler, store)

        assert await w.process_single_task("m1", ingestion_task()) == TaskStatus.CANCELLED
        assert dispatcher.delivered_events == []
        assert "m1" in transport._acked
        assert not handler.extracted.is_set()


async def test_an_internal_retry_hands_the_lease_to_the_next_delivery():
    class Transient(TaskHandler):
        async def handle(self, ctx, payload):
            raise TransientIngestionError("OBJECT_STORAGE_UNAVAILABLE", "down")

    store = InMemoryTaskRecordStore()
    await store.create(record())
    w, _, transport = worker(Transient(), store)

    assert await w.process_single_task("m1", ingestion_task()) == TaskStatus.FAILED
    stored = await store.get("task_1")
    assert (stored.state, stored.worker_id, stored.lease_until) == (
        TaskState.ACCEPTED,
        None,
        None,
    )
    assert not transport._queue.empty()  # re-queued
    assert await store.start("task_1", "w2", 60) == WriteResult.OK
