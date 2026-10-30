import json
import logging
from dataclasses import replace

import pytest
from prometheus_client import generate_latest

from app.core.callbacks.dispatcher import InMemoryCallbackDispatcher
from app.core.execution_context import AITask, TaskStatus
from app.core.logging import JsonFormatter
from app.core.metrics import (
    expired_lease_candidates,
    stream_entries_reclaimed_total,
    task_transitions_total,
)
from app.services.tasks.handlers import TaskHandler, TaskHandlerRegistry
from app.services.tasks.reclaimer import StreamReclaimer
from app.services.tasks.transport import InMemoryTaskTransport
from app.services.tasks.worker import TaskWorker


class SucceedingHandler(TaskHandler):
    async def handle(self, _ctx, _payload):
        return {"result": "ok"}


def _task() -> AITask:
    return AITask(
        task_id="task-observability",
        task_type="test.success",
        execution_id="exec-observability",
        tenant_id="tenant-observability",
        job_id="job-observability",
        asset_id="asset-observability",
        run_number=2,
        attempt_count=3,
        correlation_id="corr-observability",
        payload={},
    )


@pytest.mark.asyncio
async def test_task_transition_emits_attempt_identity_and_increments_counter(caplog):
    transport = InMemoryTaskTransport()
    registry = TaskHandlerRegistry()
    registry.register("test.success", SucceedingHandler())
    worker = TaskWorker(
        worker_id="worker-observability",
        transport=transport,
        registry=registry,
        callback_dispatcher=InMemoryCallbackDispatcher(),
    )
    task = _task()
    msg_id = await transport.publish(task)
    before = task_transitions_total.labels(outcome="completed")._value.get()

    with caplog.at_level(logging.INFO, logger="app.services.tasks.worker"):
        result = await worker.process_single_task(msg_id, task)

    assert result is TaskStatus.SUCCEEDED
    assert task_transitions_total.labels(outcome="completed")._value.get() == before + 1
    transition = next(
        record for record in caplog.records if record.msg == "ai_service_task_completed"
    )
    assert (
        transition.job_id,
        transition.task_id,
        transition.run_number,
        transition.attempt,
    ) == (
        task.job_id,
        task.task_id,
        task.run_number,
        task.attempt_count,
    )


class ReclaimClient:
    def __init__(self, task: AITask):
        self.task = task

    async def xpending_range(self, *_args, **_kwargs):
        return [
            {
                "message_id": "stream-1",
                "consumer": "worker-old",
                "time_since_delivered": 120_000,
            }
        ]

    async def xrange(self, *_args, **_kwargs):
        return [("stream-1", {"data": json.dumps(self.task.to_dict())})]


class ReclaimStore:
    async def reclaim(self, *_args, **_kwargs):
        return "lost"


@pytest.mark.asyncio
async def test_reclaimer_counts_lost_entries_and_sets_expired_lease_gauge():
    before = stream_entries_reclaimed_total.labels(outcome="lost")._value.get()
    task = replace(_task(), task_type="content.ingestion")
    reclaimer = StreamReclaimer(
        ReclaimClient(task),
        ReclaimStore(),
        group="workers",
        lease_seconds=60,
    )

    outcomes = await reclaimer.run_once()

    assert outcomes == {"lost": 1}
    assert stream_entries_reclaimed_total.labels(outcome="lost")._value.get() == before + 1
    assert expired_lease_candidates._value.get() == 1


def test_json_logs_include_only_allowlisted_transition_fields():
    record = logging.LogRecord(
        "test",
        logging.INFO,
        __file__,
        1,
        "ai_service_task_completed",
        (),
        None,
    )
    record.job_id = "job-1"
    record.task_id = "task-1"
    record.run_number = 2
    record.attempt = 3
    record.api_key = "must-not-be-serialized"

    rendered = json.loads(JsonFormatter().format(record))

    assert rendered["job_id"] == "job-1"
    assert rendered["task_id"] == "task-1"
    assert rendered["run_number"] == 2
    assert rendered["attempt"] == 3
    assert "api_key" not in rendered
    assert "must-not-be-serialized" not in json.dumps(rendered)


def test_ai_service_task_metrics_are_registered():
    payload = generate_latest()

    assert b"ai_service_task_transitions_total" in payload
    assert b"ai_service_callback_delivery_failures_total" in payload
    assert b"ai_service_stream_entries_reclaimed_total" in payload
    assert b"ai_service_expired_lease_candidates" in payload
