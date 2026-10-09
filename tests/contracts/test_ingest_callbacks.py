"""The callback body AI-service sends matches the backend contract (design B.3)."""

import pytest
from pydantic import ValidationError

from app.core.callbacks.dispatcher import backend_callback_body
from app.core.callbacks.events import CallbackEvent, CallbackEventType
from app.schemas.ingest import IngestCallbackPayload

TENANT = "6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c"


def _event(event_type, **extra):
    return CallbackEvent(
        job_id="job_contract_123",
        asset_id="asset_contract_456",
        tenant_id=TENANT,
        run_number=1,
        attempt_count=1,
        external_task_id="task_contract_001",
        correlation_id="corr_contract_999",
        event_type=event_type,
        **extra,
    )


def test_completed_event_produces_a_valid_job_completed_body():
    body = backend_callback_body(
        _event(
            CallbackEventType.COMPLETED,
            extracted_metadata={"document_type": "syllabus", "pages": 12},
            chunk_summary={"chunks_created": 36},
        )
    )
    parsed = IngestCallbackPayload.model_validate(body)
    assert parsed.event_type == "job_completed"
    assert parsed.metrics["extracted_metadata"] == {"document_type": "syllabus", "pages": 12}


def test_failed_event_produces_a_valid_job_failed_body():
    body = backend_callback_body(
        _event(CallbackEventType.FAILED, error_code="X", error_message="y", retryable=True)
    )
    parsed = IngestCallbackPayload.model_validate(body)
    assert (parsed.event_type, parsed.error_code, parsed.is_retryable) == ("job_failed", "X", True)


def test_lifecycle_events_without_a_backend_transition_produce_no_body():
    assert backend_callback_body(_event(CallbackEventType.STARTED)) is None
    assert backend_callback_body(_event(CallbackEventType.PROGRESS, progress_pct=40)) is None


BASE = {
    "event_id": "3f1c0d9e-6b0a-4c55-9d43-1c2b8f7a9e10",
    "job_id": "job-1",
    "external_task_id": "task-1",
    "run_number": 1,
    "attempt_count": 1,
    "event_type": "job_completed",
}


@pytest.mark.parametrize(
    "extra",
    [
        {"tenant_id": TENANT},
        {"user_id": "u"},
        {"role": "admin"},
        {"callback_url": "https://x"},
        {"progress_pct": 50},
    ],
)
def test_identity_destination_and_unknown_fields_are_rejected(extra):
    with pytest.raises(ValidationError):
        IngestCallbackPayload.model_validate({**BASE, **extra})


@pytest.mark.parametrize("field", ["event_id", "job_id", "external_task_id", "run_number"])
def test_required_fields(field):
    with pytest.raises(ValidationError):
        IngestCallbackPayload.model_validate({k: v for k, v in BASE.items() if k != field})


@pytest.mark.parametrize(
    "change",
    [
        {"event_id": "evt_abc"},
        {"event_type": "COMPLETED"},
        {"stage": "uploaded"},
        {"run_number": 0},
        {"metrics": "not an object"},
    ],
)
def test_wrong_values_are_rejected(change):
    with pytest.raises(ValidationError):
        IngestCallbackPayload.model_validate({**BASE, **change})


@pytest.mark.parametrize("stage", ["extracted", "chunked", "indexed"])
def test_stage_events_produce_valid_stage_progress_bodies(stage):
    body = backend_callback_body(
        _event(CallbackEventType.STAGE_COMPLETED, stage=stage, metrics={"chunks": 3})
    )
    parsed = IngestCallbackPayload.model_validate(body)
    assert (parsed.event_type, parsed.stage, parsed.metrics) == (
        "stage_progress",
        stage,
        {"chunks": 3},
    )


def test_an_unknown_stage_cannot_be_expressed():
    with pytest.raises(ValidationError):
        backend_callback_body(
            _event(CallbackEventType.STAGE_COMPLETED, stage="embedded", metrics={})
        )
