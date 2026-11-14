"""#54: an ingestion failure reaches the backend as a code and a retry flag.

The job_failed callback carries a stable failure code, whether a new attempt
may help (from the failure's category), and a fixed message. Nothing an
exception said crosses: each test plants a sentinel in a failure and checks it
reaches neither the callback body nor the dead-letter record nor the logs.
"""

import json
import logging

import pytest

from app.core.callbacks.dispatcher import (
    FAILURE_CALLBACK_MESSAGE,
    InMemoryCallbackDispatcher,
    backend_callback_body,
)
from app.core.callbacks.events import CallbackEvent, CallbackEventType
from app.core.callbacks.failure_codes import FAILURE_CODES
from app.core.execution_context import AITask
from app.services.tasks.handlers import TaskHandler, TaskHandlerRegistry
from app.services.tasks.ingestion_outcomes import (
    PermanentIngestionError,
    PipelineUnavailableError,
    TransientIngestionError,
)
from app.services.tasks.task_records import TaskRecord
from app.services.tasks.transport import InMemoryTaskTransport
from app.services.tasks.worker import TaskWorker

SENTINEL = "student-passage-c4f8"


class Raises(TaskHandler):
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def handle(self, ctx, payload):
        raise self.exc


def task() -> AITask:
    return AITask(
        task_id="task_1",
        task_type="content.ingestion",
        execution_id="exec_1",
        tenant_id="6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c",
        job_id="job_1",
        asset_id="asset_1",
        run_number=2,
        attempt_count=3,
        correlation_id="corr_1",
        payload={},
        max_attempts=1,
    )


async def fail_with(exc: Exception):
    transport = InMemoryTaskTransport()
    registry = TaskHandlerRegistry()
    registry.register("content.ingestion", Raises(exc))
    dispatcher = InMemoryCallbackDispatcher()
    worker = TaskWorker(transport=transport, registry=registry, callback_dispatcher=dispatcher)
    t = task()
    await worker.task_records.create(
        TaskRecord(
            external_task_id=t.task_id,
            tenant_id=t.tenant_id,
            job_id=t.job_id,
            run_number=t.run_number,
            attempt_count=t.attempt_count,
        )
    )
    await worker.process_single_task("msg-1", t)
    [failed] = [e for e in dispatcher.delivered_events if e.event_type == CallbackEventType.FAILED]
    return failed, transport, worker


FAILURES = [
    (RuntimeError(f"engine crashed on: {SENTINEL}"), "AI_UNCLASSIFIED_ERROR", False),
    (
        TransientIngestionError("OBJECT_STORAGE_UNAVAILABLE", f"storage said: {SENTINEL}"),
        "OBJECT_STORAGE_UNAVAILABLE",
        True,
    ),
    (PermanentIngestionError("OCR_PAGE_FAILED", f"ocr said: {SENTINEL}"), "OCR_PAGE_FAILED", False),
    (
        PipelineUnavailableError(f"OCR is not available: {SENTINEL}"),
        "INGESTION_PIPELINE_UNAVAILABLE",
        False,
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("exc", "code", "retryable"), FAILURES, ids=[f[1] for f in FAILURES])
async def test_a_failure_is_reported_by_code_and_retry_flag_never_by_its_text(
    exc, code, retryable, caplog
):
    with caplog.at_level(logging.DEBUG):
        failed, transport, _ = await fail_with(exc)

    body = backend_callback_body(failed)
    assert body is not None
    assert (body["event_type"], body["error_code"], body["is_retryable"]) == (
        "job_failed",
        code,
        retryable,
    )
    assert body["error_message"] == FAILURE_CALLBACK_MESSAGE
    assert failed.error_message is None  # the event itself carries no text
    assert SENTINEL not in json.dumps(body)
    assert SENTINEL not in json.dumps(transport.dlq_items, default=str)
    assert all(SENTINEL not in r.getMessage() for r in caplog.records)


def test_the_callback_sends_only_the_fixed_message_whatever_the_event_holds():
    """Enforced where the body is built, not only by today's one producer."""
    event = CallbackEvent(
        job_id="job_1",
        asset_id="asset_1",
        tenant_id="6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c",
        run_number=1,
        attempt_count=1,
        external_task_id="task_1",
        correlation_id="corr_1",
        event_type=CallbackEventType.FAILED,
        error_code="OCR_PAGE_FAILED",
        error_message=f"raw: {SENTINEL}",
        retryable=False,
    )

    body = backend_callback_body(event)

    assert body is not None and body["error_message"] == FAILURE_CALLBACK_MESSAGE
    assert SENTINEL not in json.dumps(body)


# --- only a listed code crosses -----------------------------------------------------------


def failed_event(code: str, retryable: bool = True) -> CallbackEvent:
    return CallbackEvent(
        job_id="job_1",
        asset_id="asset_1",
        tenant_id="6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c",
        run_number=1,
        attempt_count=1,
        external_task_id="task_1",
        correlation_id="corr_1",
        event_type=CallbackEventType.FAILED,
        error_code=code,
        retryable=retryable,
    )


@pytest.mark.parametrize("code", sorted(FAILURE_CODES))
@pytest.mark.parametrize("retryable", [True, False])
def test_every_listed_code_crosses_unchanged(code, retryable):
    body = backend_callback_body(failed_event(code, retryable))

    assert body is not None and (body["error_code"], body["is_retryable"]) == (code, retryable)


def test_a_code_outside_the_set_is_reported_as_unclassified_and_not_retried(caplog):
    with caplog.at_level(logging.WARNING):
        body = backend_callback_body(failed_event("NOT_A_LISTED_CODE", retryable=True))

    assert body is not None
    assert (body["error_code"], body["is_retryable"]) == ("AI_UNCLASSIFIED_ERROR", False)
    assert any(
        "Unlisted failure code replaced: NOT_A_LISTED_CODE" in r.getMessage()
        for r in caplog.records
    )


def test_a_malformed_code_is_neither_sent_nor_logged(caplog):
    with caplog.at_level(logging.DEBUG):
        body = backend_callback_body(failed_event(f"oops: {SENTINEL}"))

    assert body is not None and body["error_code"] == "AI_UNCLASSIFIED_ERROR"
    assert SENTINEL not in json.dumps(body)
    assert all(SENTINEL not in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_the_reconcile_copy_is_enforced_like_the_callback():
    """The worker records the same body for the backend's reconcile pull."""
    failed, _, worker = await fail_with(
        PermanentIngestionError("NOT_A_LISTED_CODE", f"raw: {SENTINEL}")
    )
    record = await worker.task_records.get("task_1")

    assert record is not None
    [recorded] = [e for e in record.events if e["event_type"] == "job_failed"]
    assert recorded == backend_callback_body(failed)  # the exact body the backend was sent
    assert (recorded["error_code"], recorded["is_retryable"]) == ("AI_UNCLASSIFIED_ERROR", False)
    assert SENTINEL not in json.dumps(record.events)
