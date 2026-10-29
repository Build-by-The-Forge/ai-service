from unittest.mock import AsyncMock, patch

import pytest

from app.core.execution_context import AITask
from app.services.tasks.transport import (
    InMemoryTaskTransport,
    RedisStreamTransport,
    TaskQueueUnavailableError,
)


@pytest.fixture
def sample_task():
    return AITask(
        task_id="task_t1",
        task_type="content.ingestion",
        execution_id="exec_t1",
        tenant_id="school_1",
        job_id="job_1",
        asset_id="asset_1",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_1",
        payload={"url": "https://example.com"},
    )


@pytest.mark.asyncio
async def test_in_memory_task_transport_publish_and_consume(sample_task):
    transport = InMemoryTaskTransport()

    msg_id = await transport.publish(sample_task)
    assert msg_id.startswith("mem_msg_")

    consumed = await transport.consume(count=1, block_ms=100)
    assert len(consumed) == 1
    consumed_msg_id, task = consumed[0]
    assert consumed_msg_id == msg_id
    assert task.task_id == sample_task.task_id

    # Acknowledge
    await transport.ack(consumed_msg_id)
    assert consumed_msg_id in transport._acked

    # DLQ routing
    dlq_id = await transport.send_to_dlq(sample_task, error_reason="Permanent parsing error")
    assert dlq_id.startswith("dlq_msg_")
    assert len(transport.dlq_items) == 1
    assert transport.dlq_items[0]["error_reason"] == "Permanent parsing error"


@pytest.mark.asyncio
async def test_redis_stream_transport_fails_closed_without_redis(sample_task):
    """A task published while Redis is down must fail the dispatch, never
    sit in this process's memory where no worker can see it."""
    transport = RedisStreamTransport()

    with patch("app.services.tasks.transport.get_redis_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.xadd.side_effect = ConnectionError("Redis unavailable")
        mock_client.xgroup_create.side_effect = ConnectionError("Redis unavailable")
        mock_client.xreadgroup.side_effect = ConnectionError("Redis unavailable")
        mock_client.xack.side_effect = ConnectionError("Redis unavailable")
        mock_get_client.return_value = mock_client

        with pytest.raises(TaskQueueUnavailableError):
            await transport.publish(sample_task)
        with pytest.raises(TaskQueueUnavailableError):
            await transport.consume(count=1, block_ms=100)
        # A failed acknowledgement leaves the entry pending (the reclaim
        # decides it later); it does not fail the caller.
        await transport.ack("1-0")
        with pytest.raises(TaskQueueUnavailableError):
            await transport.send_to_dlq(sample_task, error_reason="x")
