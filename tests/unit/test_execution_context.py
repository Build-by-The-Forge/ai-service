import time
from app.core.execution_context import AIExecutionContext, AITask, TaskStatus


def test_ai_task_envelope_round_trip():
    task = AITask(
        task_id="task_123",
        task_type="content.ingestion",
        execution_id="exec_456",
        tenant_id="school_futa",
        job_id="job_789",
        asset_id="asset_001",
        run_number=2,
        attempt_count=1,
        correlation_id="corr_999",
        payload={"storage_uri": "s3://bucket/pack.pdf", "mime_type": "application/pdf"},
        deadline=time.time() + 60,
        max_attempts=3,
    )

    data = task.to_dict()
    assert data["task_id"] == "task_123"
    assert data["task_type"] == "content.ingestion"
    assert data["run_number"] == 2
    assert not task.is_expired()

    restored = AITask.from_dict(data)
    assert restored.task_id == task.task_id
    assert restored.execution_id == task.execution_id
    assert restored.payload == task.payload


def test_ai_task_expiration():
    expired_task = AITask(
        task_id="task_exp",
        task_type="content.ingestion",
        execution_id="exec_exp",
        tenant_id="school_1",
        job_id="job_1",
        asset_id="asset_1",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_1",
        payload={},
        deadline=time.time() - 10,
    )
    assert expired_task.is_expired() is True


def test_ai_execution_context_cancellation_and_metadata():
    task = AITask(
        task_id="task_ctx",
        task_type="content.ingestion",
        execution_id="exec_ctx",
        tenant_id="school_1",
        job_id="job_1",
        asset_id="asset_1",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_1",
        payload={},
    )

    ctx = AIExecutionContext.from_task(task, actor_context={"user": "admin"})
    assert ctx.execution_id == "exec_ctx"
    assert ctx.tenant_id == "school_1"
    assert ctx.is_cancelled() is False

    ctx.set_metadata("key1", "val1")
    assert ctx.get_metadata("key1") == "val1"

    ctx.cancel("Operator aborted job")
    assert ctx.is_cancelled() is True
    assert ctx.cancellation_reason == "Operator aborted job"


def test_task_status_enum_values():
    assert TaskStatus.ACCEPTED == "ACCEPTED"
    assert TaskStatus.QUEUED == "QUEUED"
    assert TaskStatus.RUNNING == "RUNNING"
    assert TaskStatus.SUCCEEDED == "SUCCEEDED"
    assert TaskStatus.FAILED == "FAILED"
    assert TaskStatus.CANCELLED == "CANCELLED"
    assert TaskStatus.EXPIRED == "EXPIRED"
    assert TaskStatus.DEAD_LETTER == "DEAD_LETTER"
