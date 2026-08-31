import pytest
from app.services.tasks.dispatcher import TaskDispatcher
from app.services.tasks.transport import InMemoryTaskTransport


@pytest.mark.asyncio
async def test_task_dispatcher_dispatch_ingest():
    transport = InMemoryTaskTransport()
    dispatcher = TaskDispatcher(transport=transport)

    task = await dispatcher.dispatch_ingest(
        job_id="job_dispatch_1",
        asset_id="asset_dispatch_1",
        tenant_id="school_1",
        correlation_id="corr_dispatch_1",
        payload={"storage_uri": "s3://pack.pdf", "mime_type": "application/pdf"},
        run_number=1,
        attempt_count=1,
    )

    assert task.task_type == "content.ingestion"
    assert task.job_id == "job_dispatch_1"
    assert task.task_id.startswith("task_")
    assert task.execution_id.startswith("exec_")

    # Verify task arrived in transport
    consumed = await transport.consume(count=1, block_ms=100)
    assert len(consumed) == 1
    assert consumed[0][1].task_id == task.task_id
