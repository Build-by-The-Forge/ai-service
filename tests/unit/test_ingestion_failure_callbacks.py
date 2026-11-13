"""#54: an ingestion failure reaches the backend as a code and a retry flag.

The job_failed callback carries a stable failure code, whether a new attempt
may help (from the failure's category), and a fixed message. Nothing an
exception said crosses: each test plants a sentinel in a failure and checks it
reaches neither the callback body nor the dead-letter record nor the logs.
"""

import json
import logging
import re
from pathlib import Path

import pytest

from app.core.callbacks.dispatcher import (
    FAILURE_CALLBACK_MESSAGE,
    InMemoryCallbackDispatcher,
    backend_callback_body,
)
from app.core.callbacks.events import CallbackEvent, CallbackEventType
from app.core.execution_context import AITask
from app.services.tasks.handlers import TaskHandler, TaskHandlerRegistry
from app.services.tasks.ingestion_outcomes import (
    FAILURE_CODES,
    PermanentIngestionError,
    PipelineUnavailableError,
    TransientIngestionError,
)
from app.services.tasks.task_records import TaskRecord
from app.services.tasks.transport import InMemoryTaskTransport
from app.services.tasks.worker import TaskWorker

SENTINEL = "student-passage-c4f8"
APP = Path(__file__).resolve().parents[2] / "app"


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
    return failed, transport


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
        failed, transport = await fail_with(exc)

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


def _codes_raised_in_app() -> set[str]:
    codes: set[str] = set()
    for path in APP.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        # A failure built with a literal code, on the same line or the next.
        codes.update(re.findall(r'IngestionError\(\s*"([A-Z][A-Z_]+)"', source))
        # A code the worker reports itself.
        if path.name == "worker.py":
            codes.update(re.findall(r'error_code="([A-Z][A-Z_]+)"', source))
            codes.update(re.findall(r'"(AI_[A-Z_]+|UNREGISTERED_HANDLER)"', source))
        # A module constant naming a code (PIPELINE_UNAVAILABLE, REFERENCE_REJECTED).
        if path.parent.name in ("tasks", "ingestion"):
            codes.update(re.findall(r'^[A-Z_]+ = "([A-Z][A-Z_]+)"$', source, flags=re.M))
    return codes


def test_every_failure_code_raised_in_app_is_in_the_closed_set():
    raised = _codes_raised_in_app()

    assert raised, "the scan found no codes"
    assert (
        raised <= FAILURE_CODES
    ), f"codes missing from FAILURE_CODES: {sorted(raised - FAILURE_CODES)}"
