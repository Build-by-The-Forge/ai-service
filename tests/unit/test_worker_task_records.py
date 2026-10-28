"""The worker records each backend-bound ingestion event for reconciliation (M3-09)."""

from app.core.callbacks.dispatcher import InMemoryCallbackDispatcher
from app.core.execution_context import AITask, TaskStatus
from app.services.tasks.handlers import (
    TaskHandler,
    TaskHandlerRegistry,
)
from app.services.tasks.ingestion_outcomes import PipelineUnavailableError
from app.services.tasks.task_records import (
    InMemoryTaskRecordStore,
    TaskRecordUnavailableError,
    TaskState,
)
from app.services.tasks.transport import InMemoryTaskTransport
from app.services.tasks.worker import TaskWorker
from tests.unit.test_task_records import TENANT, record


class Staged(TaskHandler):
    async def handle(self, ctx, payload):
        await ctx.stage_reporter.report("extracted", {"characters": 9, "pages": 1, "method": "ocr"})
        await ctx.stage_reporter.report("chunked", {"chunks": 2})
        await ctx.stage_reporter.report("indexed", {"indexed_chunks": 2, "index": "pgvector"})
        return {}


class PipelineUnavailable(TaskHandler):
    """Fails as an ingestion task does while the pipeline is unavailable."""

    async def handle(self, ctx, payload):
        raise PipelineUnavailableError("not available")


def ingestion_task(task_id="task_1"):
    return AITask(
        task_id=task_id,
        task_type="content.ingestion",
        execution_id="exec_1",
        tenant_id=TENANT,
        job_id="job_1",
        asset_id="asset_1",
        run_number=1,
        attempt_count=2,
        correlation_id="corr-1",
        payload={},
    )


def worker(handler, store, dispatcher=None, task_type="content.ingestion"):
    registry = TaskHandlerRegistry()
    registry.register(task_type, handler)
    return TaskWorker(
        transport=InMemoryTaskTransport(),
        registry=registry,
        callback_dispatcher=dispatcher or InMemoryCallbackDispatcher(),
        task_records=store,
    )


async def test_a_successful_task_records_every_reported_event():
    store = InMemoryTaskRecordStore()
    await store.create(record())
    assert (
        await worker(Staged(), store).process_single_task("m1", ingestion_task())
        == TaskStatus.SUCCEEDED
    )
    stored = await store.get("task_1")
    assert stored.state == TaskState.SUCCEEDED
    assert [(e["event_type"], e.get("stage")) for e in stored.events] == [
        ("stage_progress", "extracted"),
        ("stage_progress", "chunked"),
        ("stage_progress", "indexed"),
        ("job_completed", None),
    ]
    assert {(e["run_number"], e["attempt_count"]) for e in stored.events} == {(1, 2)}


async def test_events_are_recorded_even_when_delivery_fails():
    store = InMemoryTaskRecordStore()
    await store.create(record())
    dispatcher = InMemoryCallbackDispatcher()
    dispatcher.simulate_failure(status_code=500)
    await worker(PipelineUnavailable(), store, dispatcher).process_single_task(
        "m1", ingestion_task()
    )
    stored = await store.get("task_1")
    assert stored.state == TaskState.FAILED
    assert [(e["event_type"], e["error_code"]) for e in stored.events] == [
        ("job_failed", "INGESTION_PIPELINE_UNAVAILABLE")
    ]


async def test_a_record_store_outage_does_not_fail_the_task():
    class Down(InMemoryTaskRecordStore):
        async def start(self, *args, **kwargs):
            raise TaskRecordUnavailableError("down")

        async def record_event(self, *args, **kwargs):
            raise TaskRecordUnavailableError("down")

    store = Down()
    await InMemoryTaskRecordStore.create(store, record())
    status = await worker(Staged(), store).process_single_task("m1", ingestion_task())
    assert status == TaskStatus.SUCCEEDED


async def test_non_ingestion_tasks_are_not_recorded():
    class Succeeds(TaskHandler):
        async def handle(self, ctx, payload):
            return {}

    store = InMemoryTaskRecordStore()
    task = AITask(
        task_id="agent_1",
        task_type="agent.execution",
        execution_id="e",
        tenant_id=TENANT,
        job_id="j",
        asset_id="a",
        run_number=1,
        attempt_count=1,
        correlation_id="c",
        payload={},
    )
    await worker(Succeeds(), store, task_type="agent.execution").process_single_task("m1", task)
    assert store.records == {}
