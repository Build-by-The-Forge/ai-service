"""Worker lifecycle callbacks (M3-08).

Callbacks always carry the backend's attempt identity; ingestion completes
only after the indexed stage was reported; failures are classified.
"""

import pytest

from app.core.callbacks.dispatcher import InMemoryCallbackDispatcher
from app.core.callbacks.events import CallbackDeliveryResult, CallbackEventType
from app.core.execution_context import AITask, TaskStatus
from app.services.tasks.handlers import TaskHandler, TaskHandlerRegistry
from app.services.tasks.ingestion_outcomes import (
    PermanentIngestionError,
    StageReporter,
    TransientIngestionError,
)
from app.services.tasks.task_records import TaskRecord
from app.services.tasks.transport import InMemoryTaskTransport
from app.services.tasks.worker import TaskWorker

EVIDENCE = {
    "extracted": {"characters": 900, "pages": 3, "method": "native_text"},
    "chunked": {"chunks": 4},
    "indexed": {"indexed_chunks": 4, "index": "pgvector"},
}


class StagedPipeline(TaskHandler):
    """A pipeline that really reports each stage before finishing."""

    def __init__(self, stages=("extracted", "chunked", "indexed")):
        self.stages = stages

    async def handle(self, ctx, payload):
        for stage in self.stages:
            await ctx.stage_reporter.report(stage, EVIDENCE[stage])
        return {"extracted_metadata": {"title": "Physics 101"}, "chunk_summary": {"chunks": 4}}


class Raises(TaskHandler):
    def __init__(self, exc):
        self.exc = exc

    async def handle(self, ctx, payload):
        raise self.exc


def task(task_type="content.ingestion", **kwargs):
    fields = dict(
        task_id="task_1",
        task_type=task_type,
        execution_id="exec_1",
        tenant_id="6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c",
        job_id="job_1",
        asset_id="asset_1",
        run_number=2,
        attempt_count=3,
        correlation_id="corr_1",
        payload={},
    )
    fields.update(kwargs)
    return AITask(**fields)


def worker_with(handler, task_type="content.ingestion", dispatcher=None):
    transport = InMemoryTaskTransport()
    registry = TaskHandlerRegistry()
    registry.register(task_type, handler)
    dispatcher = dispatcher or InMemoryCallbackDispatcher()
    return (
        TaskWorker(transport=transport, registry=registry, callback_dispatcher=dispatcher),
        transport,
        dispatcher,
    )


async def run(worker, transport, t):
    """Process a task as if just consumed: the queue then holds only re-queues.
    Its task record exists, as dispatch always creates it before publishing."""
    await worker.task_records.create(
        TaskRecord(
            external_task_id=t.task_id,
            tenant_id=t.tenant_id,
            job_id=t.job_id,
            run_number=t.run_number,
            attempt_count=t.attempt_count,
        )
    )
    msg_id = f"msg-{t.delivery_attempt}"
    return msg_id, await worker.process_single_task(msg_id, t)


def kinds(dispatcher):
    return [(e.event_type, e.stage) for e in dispatcher.delivered_events]


# --- success requires the indexed stage -------------------------------------------------


async def test_a_staged_pipeline_reports_each_stage_then_completes():
    worker, transport, dispatcher = worker_with(StagedPipeline())
    _, status = await run(worker, transport, task())
    assert status == TaskStatus.SUCCEEDED
    assert kinds(dispatcher) == [
        (CallbackEventType.STARTED, None),
        (CallbackEventType.STAGE_COMPLETED, "extracted"),
        (CallbackEventType.STAGE_COMPLETED, "chunked"),
        (CallbackEventType.STAGE_COMPLETED, "indexed"),
        (CallbackEventType.COMPLETED, None),
    ]
    stage = dispatcher.delivered_events[1]
    assert stage.metrics == EVIDENCE["extracted"]
    # Every event carries the backend's attempt identity unchanged.
    assert {
        (e.run_number, e.attempt_count, e.external_task_id) for e in dispatcher.delivered_events
    } == {(2, 3, "task_1")}


@pytest.mark.parametrize("stages", [(), ("extracted",), ("extracted", "chunked")])
async def test_finishing_without_the_indexed_stage_is_not_completion(stages):
    worker, transport, dispatcher = worker_with(StagedPipeline(stages))
    _, status = await run(worker, transport, task())
    assert status == TaskStatus.FAILED
    last = dispatcher.delivered_events[-1]
    assert (last.event_type, last.error_code, last.retryable) == (
        CallbackEventType.FAILED,
        "INGESTION_INCOMPLETE",
        False,
    )
    assert CallbackEventType.COMPLETED not in [e.event_type for e in dispatcher.delivered_events]


async def test_stages_out_of_order_stop_the_pipeline():
    worker, transport, dispatcher = worker_with(StagedPipeline(("chunked",)))
    _, status = await run(worker, transport, task())
    assert status == TaskStatus.FAILED
    assert dispatcher.delivered_events[-1].error_code == "STAGE_OUT_OF_ORDER"
    assert CallbackEventType.STAGE_COMPLETED not in [
        e.event_type for e in dispatcher.delivered_events
    ]


# --- failure classification -----------------------------------------------------------------


async def test_permanent_failures_are_not_retried():
    worker, transport, dispatcher = worker_with(
        Raises(PermanentIngestionError("UNSUPPORTED_FORMAT", "no"))
    )
    _, status = await run(worker, transport, task(max_attempts=3))
    assert status == TaskStatus.FAILED
    failed = dispatcher.delivered_events[-1]
    assert (failed.error_code, failed.retryable) == ("UNSUPPORTED_FORMAT", False)
    assert transport._queue.empty()


async def test_transient_failures_retry_the_same_backend_attempt():
    worker, transport, dispatcher = worker_with(
        Raises(TransientIngestionError("STORAGE_TIMEOUT", "slow"))
    )
    _, status = await run(worker, transport, task(max_attempts=2))
    assert status == TaskStatus.FAILED
    _, requeued = transport._queue.get_nowait()
    assert (requeued.attempt_count, requeued.run_number, requeued.delivery_attempt) == (3, 2, 2)
    assert CallbackEventType.FAILED not in [e.event_type for e in dispatcher.delivered_events]

    # Last delivery: reported as retryable for a new backend attempt.
    _, status = await run(worker, transport, requeued)
    assert status == TaskStatus.DEAD_LETTER
    failed = dispatcher.delivered_events[-1]
    assert (failed.error_code, failed.retryable, failed.attempt_count) == (
        "STORAGE_TIMEOUT",
        True,
        3,
    )


async def test_unclassified_errors_end_non_retryable():
    worker, transport, dispatcher = worker_with(
        Raises(RuntimeError("Fatal OCR engine crash")), "crash.task"
    )
    _, status = await run(worker, transport, task("crash.task", max_attempts=1))
    assert status == TaskStatus.DEAD_LETTER
    failed = dispatcher.delivered_events[-1]
    assert (failed.error_code, failed.retryable) == ("AI_UNCLASSIFIED_ERROR", False)
    assert "Fatal OCR engine crash" in failed.error_message


# --- stage delivery --------------------------------------------------------------------------


class RefusingDispatcher(InMemoryCallbackDispatcher):
    """Refuses stage events with a given status; accepts everything else."""

    def __init__(self, status_code: int, retryable: bool) -> None:
        super().__init__()
        self.status_code, self.retryable = status_code, retryable

    async def dispatch(self, event):
        if event.event_type == CallbackEventType.STAGE_COMPLETED:
            return CallbackDeliveryResult(
                success=False,
                status_code=self.status_code,
                retryable=self.retryable,
                event_id=event.event_id,
            )
        return await super().dispatch(event)


async def test_an_undeliverable_stage_stops_the_pipeline_and_retries_when_transient():
    dispatcher = RefusingDispatcher(503, retryable=True)
    worker, transport, _ = worker_with(StagedPipeline(), dispatcher=dispatcher)
    _, status = await run(worker, transport, task(max_attempts=2))
    assert status == TaskStatus.FAILED  # re-queued for another delivery
    assert not transport._queue.empty()
    assert CallbackEventType.COMPLETED not in [e.event_type for e in dispatcher.delivered_events]


async def test_a_refused_stage_is_a_permanent_failure():
    dispatcher = RefusingDispatcher(409, retryable=False)
    worker, transport, _ = worker_with(StagedPipeline(), dispatcher=dispatcher)
    _, status = await run(worker, transport, task(max_attempts=3))
    assert status == TaskStatus.FAILED
    assert transport._queue.empty()
    assert dispatcher.delivered_events[-1].error_code == "CALLBACK_DELIVERY_FAILED"


async def test_callback_failure_does_not_fail_a_non_ingestion_task():
    class Succeeds(TaskHandler):
        async def handle(self, ctx, payload):
            return {"ok": True}

    dispatcher = InMemoryCallbackDispatcher()
    dispatcher.simulate_failure(status_code=500)
    worker, transport, _ = worker_with(Succeeds(), "agent.execution", dispatcher=dispatcher)
    msg_id, status = await run(worker, transport, task("agent.execution"))
    assert status == TaskStatus.SUCCEEDED
    assert msg_id in transport._acked


async def test_stage_reporter_rejects_repeats_and_skips():
    delivered = []

    async def deliver(stage, metrics):
        delivered.append(stage)
        return True, False

    reporter = StageReporter(deliver)
    await reporter.report("extracted", {})
    with pytest.raises(PermanentIngestionError):
        await reporter.report("extracted", {})
    with pytest.raises(PermanentIngestionError):
        await reporter.report("indexed", {})
    assert delivered == ["extracted"] and not reporter.indexed
