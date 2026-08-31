from app.core.callbacks.events import CallbackEvent, CallbackEventType
from app.core.callbacks.signer import CallbackSigner, canonical_serialize
from app.schemas.ingest import IngestCallbackPayload


def test_callback_event_validates_against_backend_payload_schema():
    event = CallbackEvent(
        job_id="job_contract_123",
        asset_id="asset_contract_456",
        tenant_id="school_contract_789",
        run_number=1,
        attempt_count=1,
        external_task_id="task_contract_001",
        correlation_id="corr_contract_999",
        event_type=CallbackEventType.COMPLETED,
        progress_pct=100,
        extracted_metadata={"document_type": "syllabus", "pages": 12},
        chunk_summary={"chunks_created": 36, "vectors_indexed": 36},
    )

    # Validate against IngestCallbackPayload Pydantic schema
    validated = IngestCallbackPayload.model_validate(event.to_dict())
    assert validated.job_id == "job_contract_123"
    assert validated.event_type.value == "COMPLETED"
    assert validated.progress_pct == 100
    assert validated.extracted_metadata == {"document_type": "syllabus", "pages": 12}


def test_all_lifecycle_events_conform_to_contract():
    events = [
        CallbackEvent(
            job_id="job_1",
            asset_id="asset_1",
            tenant_id="school_1",
            run_number=1,
            attempt_count=1,
            external_task_id="task_1",
            correlation_id="corr_1",
            event_type=CallbackEventType.STARTED,
            progress_pct=0,
        ),
        CallbackEvent(
            job_id="job_1",
            asset_id="asset_1",
            tenant_id="school_1",
            run_number=1,
            attempt_count=1,
            external_task_id="task_1",
            correlation_id="corr_1",
            event_type=CallbackEventType.PROGRESS,
            progress_pct=50,
        ),
        CallbackEvent(
            job_id="job_1",
            asset_id="asset_1",
            tenant_id="school_1",
            run_number=1,
            attempt_count=1,
            external_task_id="task_1",
            correlation_id="corr_1",
            event_type=CallbackEventType.COMPLETED,
            progress_pct=100,
            extracted_metadata={"title": "Exam Prep"},
        ),
        CallbackEvent(
            job_id="job_1",
            asset_id="asset_1",
            tenant_id="school_1",
            run_number=1,
            attempt_count=1,
            external_task_id="task_1",
            correlation_id="corr_1",
            event_type=CallbackEventType.FAILED,
            error_code="UNSUPPORTED_MIME",
            error_message="MIME type image/svg+xml not supported",
            retryable=False,
        ),
    ]

    for ev in events:
        validated = IngestCallbackPayload.model_validate(ev.to_dict())
        assert validated.job_id == "job_1"
        assert validated.event_type.value == ev.event_type.value


def test_callback_canonical_signing_contract():
    secret = "build-contract-shared-secret"
    signer = CallbackSigner(secret)

    event = CallbackEvent(
        job_id="job_sign_1",
        asset_id="asset_sign_1",
        tenant_id="school_1",
        run_number=1,
        attempt_count=1,
        external_task_id="task_sign_1",
        correlation_id="corr_sign_1",
        event_type=CallbackEventType.COMPLETED,
        progress_pct=100,
    )

    payload_bytes = canonical_serialize(event)
    timestamp = 1788158900
    signature = signer.sign(payload_bytes, timestamp)

    assert signer.verify(signature, payload_bytes, timestamp) is True
