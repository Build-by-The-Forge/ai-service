import pytest

from app.core.callbacks.dispatcher import InMemoryCallbackDispatcher
from app.core.callbacks.events import CallbackEventType
from app.core.execution_context import AITask, TaskStatus
from app.services.tasks.handlers import TaskHandler, TaskHandlerRegistry
from app.services.tasks.transport import InMemoryTaskTransport
from app.services.tasks.worker import TaskWorker


class IngestionTestHandler(TaskHandler):
    async def handle(self, ctx, payload):
        return {
            "status": "COMPLETED",
            "extracted_metadata": {"title": "Physics 101 Notes"},
            "chunk_summary": {"total_chunks": 42},
        }


class CrashHandler(TaskHandler):
    async def handle(self, ctx, payload):
        raise RuntimeError("Fatal OCR engine crash")


@pytest.mark.asyncio
async def test_worker_emits_started_and_completed_callbacks():
    transport = InMemoryTaskTransport()
    registry = TaskHandlerRegistry()
    registry.register("content.ingestion", IngestionTestHandler())
    cb_dispatcher = InMemoryCallbackDispatcher()

    worker = TaskWorker(
        transport=transport,
        registry=registry,
        callback_dispatcher=cb_dispatcher,
    )

    task = AITask(
        task_id="task_cb_flow",
        task_type="content.ingestion",
        execution_id="exec_cb_flow",
        tenant_id="school_futa",
        job_id="job_cb_flow",
        asset_id="asset_cb_flow",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_cb_flow",
        payload={"callback_url": "https://backend.build.edu/api/v1/callbacks/ingest"},
    )

    msg_id = await transport.publish(task)
    status = await worker.process_single_task(msg_id, task)

    assert status == TaskStatus.SUCCEEDED

    # Verify callbacks emitted: STARTED then COMPLETED
    events = cb_dispatcher.delivered_events
    assert len(events) == 2

    started_evt = events[0]
    assert started_evt.event_type == CallbackEventType.STARTED
    assert started_evt.job_id == "job_cb_flow"
    assert started_evt.tenant_id == "school_futa"
    assert started_evt.progress_pct == 0

    completed_evt = events[1]
    assert completed_evt.event_type == CallbackEventType.COMPLETED
    assert completed_evt.job_id == "job_cb_flow"
    assert completed_evt.tenant_id == "school_futa"
    assert completed_evt.progress_pct == 100
    assert completed_evt.extracted_metadata == {"title": "Physics 101 Notes"}
    assert completed_evt.chunk_summary == {"total_chunks": 42}


@pytest.mark.asyncio
async def test_worker_emits_failed_callback_on_terminal_failure():
    transport = InMemoryTaskTransport()
    registry = TaskHandlerRegistry()
    registry.register("crash.task", CrashHandler())
    cb_dispatcher = InMemoryCallbackDispatcher()

    worker = TaskWorker(
        transport=transport,
        registry=registry,
        callback_dispatcher=cb_dispatcher,
    )

    # Task on max attempt (attempt_count=1, max_attempts=1) -> will fail terminally to DLQ
    task = AITask(
        task_id="task_fail_dlq",
        task_type="crash.task",
        execution_id="exec_fail_dlq",
        tenant_id="school_1",
        job_id="job_fail_1",
        asset_id="asset_fail_1",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_fail_1",
        payload={"callback_url": "https://backend.build.edu/api/v1/callbacks/ingest"},
        max_attempts=1,
    )

    msg_id = await transport.publish(task)
    status = await worker.process_single_task(msg_id, task)

    assert status == TaskStatus.DEAD_LETTER

    events = cb_dispatcher.delivered_events
    assert len(events) == 2
    assert events[0].event_type == CallbackEventType.STARTED
    assert events[1].event_type == CallbackEventType.FAILED
    assert events[1].error_code == "MAX_ATTEMPTS_EXCEEDED"
    assert "Fatal OCR engine crash" in events[1].error_message


@pytest.mark.asyncio
async def test_worker_callback_failure_does_not_fail_successful_task():
    transport = InMemoryTaskTransport()
    registry = TaskHandlerRegistry()
    registry.register("content.ingestion", IngestionTestHandler())

    # Simulate failing callback endpoint (e.g. backend returning 500 error)
    cb_dispatcher = InMemoryCallbackDispatcher()
    cb_dispatcher.simulate_failure(status_code=500)

    worker = TaskWorker(
        transport=transport,
        registry=registry,
        callback_dispatcher=cb_dispatcher,
    )

    task = AITask(
        task_id="task_cb_resilience",
        task_type="content.ingestion",
        execution_id="exec_cb_resilience",
        tenant_id="school_1",
        job_id="job_res_1",
        asset_id="asset_res_1",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_res_1",
        payload={"callback_url": "https://backend.build.edu/api/v1/callbacks/ingest"},
    )

    msg_id = await transport.publish(task)
    status = await worker.process_single_task(msg_id, task)

    # Task execution must REMAIN SUCCEEDED despite callback delivery failure
    assert status == TaskStatus.SUCCEEDED
    assert msg_id in transport._acked
