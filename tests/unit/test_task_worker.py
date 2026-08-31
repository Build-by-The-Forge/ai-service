import pytest

from app.core.execution_context import AITask, TaskStatus
from app.services.tasks.handlers import TaskHandler, TaskHandlerRegistry
from app.services.tasks.transport import InMemoryTaskTransport
from app.services.tasks.worker import TaskWorker


class SucceedingHandler(TaskHandler):
    async def handle(self, ctx, payload):
        return {"status": "ok"}


class FailingHandler(TaskHandler):
    async def handle(self, ctx, payload):
        raise ValueError("Simulated parsing crash")


@pytest.mark.asyncio
async def test_task_worker_processes_succeeding_task():
    transport = InMemoryTaskTransport()
    registry = TaskHandlerRegistry()
    registry.register("test.success", SucceedingHandler())

    worker = TaskWorker(worker_id="test_worker_1", transport=transport, registry=registry)

    task = AITask(
        task_id="task_s1",
        task_type="test.success",
        execution_id="exec_s1",
        tenant_id="school_1",
        job_id="job_1",
        asset_id="asset_1",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_1",
        payload={"foo": "bar"},
    )

    msg_id = await transport.publish(task)
    status = await worker.process_single_task(msg_id, task)

    assert status == TaskStatus.SUCCEEDED
    assert msg_id in transport._acked


@pytest.mark.asyncio
async def test_task_worker_unregistered_handler_routes_to_dlq():
    transport = InMemoryTaskTransport()
    registry = TaskHandlerRegistry()  # Empty, no handler registered for 'unknown.task'

    worker = TaskWorker(transport=transport, registry=registry)

    task = AITask(
        task_id="task_unreg",
        task_type="unknown.task",
        execution_id="exec_unreg",
        tenant_id="school_1",
        job_id="job_1",
        asset_id="asset_1",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_1",
        payload={},
    )

    msg_id = await transport.publish(task)
    status = await worker.process_single_task(msg_id, task)

    assert status == TaskStatus.DEAD_LETTER
    assert len(transport.dlq_items) == 1
    assert "No handler registered" in transport.dlq_items[0]["error_reason"]


@pytest.mark.asyncio
async def test_task_worker_retries_then_dead_letters():
    transport = InMemoryTaskTransport()
    registry = TaskHandlerRegistry()
    registry.register("test.fail", FailingHandler())

    worker = TaskWorker(transport=transport, registry=registry)

    # Task on attempt 1 (max_attempts = 2) -> Should retry
    task_attempt_1 = AITask(
        task_id="task_f1",
        task_type="test.fail",
        execution_id="exec_f1",
        tenant_id="school_1",
        job_id="job_1",
        asset_id="asset_1",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_1",
        payload={},
        max_attempts=2,
    )

    msg_id1 = await transport.publish(task_attempt_1)
    status1 = await worker.process_single_task(msg_id1, task_attempt_1)

    assert status1 == TaskStatus.FAILED

    # Task on attempt 2 (max_attempts = 2) -> Should route to DLQ
    task_attempt_2 = AITask(
        task_id="task_f1",
        task_type="test.fail",
        execution_id="exec_f1",
        tenant_id="school_1",
        job_id="job_1",
        asset_id="asset_1",
        run_number=1,
        attempt_count=2,
        correlation_id="corr_1",
        payload={},
        max_attempts=2,
    )

    msg_id2 = await transport.publish(task_attempt_2)
    status2 = await worker.process_single_task(msg_id2, task_attempt_2)

    assert status2 == TaskStatus.DEAD_LETTER
    assert len(transport.dlq_items) == 1
    assert "Exceeded max attempts" in transport.dlq_items[0]["error_reason"]


@pytest.mark.asyncio
async def test_task_worker_run_once_loop():
    transport = InMemoryTaskTransport()
    registry = TaskHandlerRegistry()
    registry.register("content.ingestion", SucceedingHandler())

    worker = TaskWorker(transport=transport, registry=registry)

    # Empty run_once returns None
    result = await worker.run_once(block_ms=10)
    assert result is None

    # Publish task and run_once processes it
    task = AITask(
        task_id="task_ingest_loop",
        task_type="content.ingestion",
        execution_id="exec_loop",
        tenant_id="school_1",
        job_id="job_1",
        asset_id="asset_1",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_1",
        payload={"storage_uri": "s3://pack.pdf"},
    )
    await transport.publish(task)

    result2 = await worker.run_once(block_ms=100)
    assert result2 == TaskStatus.SUCCEEDED
