import asyncio
import time

import pytest

from app.core.execution_context import AITask, TaskStatus
from app.services.tasks.handlers import TaskHandler, TaskHandlerRegistry
from app.services.tasks.heartbeat import HeartbeatManager
from app.services.tasks.transport import InMemoryTaskTransport
from app.services.tasks.worker import TaskWorker


class LongRunningCancellableHandler(TaskHandler):
    async def handle(self, ctx, payload):
        for _ in range(10):
            if ctx.is_cancelled():
                raise asyncio.CancelledError("Cancelled in handler")
            await asyncio.sleep(0.01)
        return {"status": "finished"}


def test_heartbeat_manager_lease_expiry():
    mgr = HeartbeatManager()
    mgr.record_heartbeat(execution_id="exec_hb_1", worker_id="w1", lease_seconds=1)

    assert mgr.is_lease_valid("exec_hb_1") is True
    assert mgr.is_lease_valid("non_existent") is False

    # Simulate expiry
    lease = mgr.get_lease("exec_hb_1")
    lease.lease_until = time.time() - 5
    assert mgr.is_lease_valid("exec_hb_1") is False

    mgr.release("exec_hb_1")
    assert mgr.get_lease("exec_hb_1") is None


@pytest.mark.asyncio
async def test_worker_cooperative_cancellation():
    transport = InMemoryTaskTransport()
    registry = TaskHandlerRegistry()
    registry.register("test.long", LongRunningCancellableHandler())

    worker = TaskWorker(transport=transport, registry=registry)

    task = AITask(
        task_id="task_cancel_1",
        task_type="test.long",
        execution_id="exec_cancel_1",
        tenant_id="school_1",
        job_id="job_1",
        asset_id="asset_1",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_1",
        payload={},
    )

    msg_id = await transport.publish(task)

    # Launch task in background and trigger cancellation
    async def cancel_soon():
        await asyncio.sleep(0.02)
        worker.cancel_task("exec_cancel_1", reason="User requested abort")

    asyncio.create_task(cancel_soon())

    status = await worker.process_single_task(msg_id, task)
    assert status == TaskStatus.CANCELLED
