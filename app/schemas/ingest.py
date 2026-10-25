import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContentAuthority(str, Enum):
    REGULATORY = "REGULATORY"
    INSTITUTIONAL = "INSTITUTIONAL"
    APPROVED_REFERENCE = "APPROVED_REFERENCE"
    EXTERNAL = "EXTERNAL"
    USER_PROVIDED = "USER_PROVIDED"
    AI_DERIVED = "AI_DERIVED"


class DispatchStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    DUPLICATE = "DUPLICATE"


class IngestDispatchRequest(BaseModel):
    """Signed dispatch body (design B.2).

    It names no tenant, user or callback destination: the tenant comes only
    from the verified context and callbacks go only to BACKEND_CALLBACK_URL.
    Unknown fields are rejected (422) rather than ignored.
    """

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    job_id: str = Field(..., min_length=1, description="Backend Ingestion Job identifier")
    asset_id: str = Field(..., min_length=1, description="Content Asset identifier")
    source_id: str = Field(..., min_length=1, description="Content Source identifier")
    storage_uri: str = Field(..., min_length=1, description="URI or path where raw asset is stored")
    mime_type: str = Field(..., min_length=1, description="MIME type of the asset")
    content_sha256: str = Field(
        ..., min_length=64, max_length=64, description="Hex-encoded SHA-256 hash of content"
    )
    # Optional until build-backend sends it (P3-1 PR 4): when present, the
    # fetched object's size must equal it, as its SHA-256 must.
    file_size_bytes: int | None = Field(
        default=None, ge=1, description="Size in bytes of the uploaded content"
    )
    run_number: int = Field(default=1, ge=1, description="Execution run number")
    attempt_count: int = Field(default=1, ge=1, description="Execution attempt count")
    correlation_id: str = Field(..., min_length=1, description="Distributed tracing correlation ID")
    idempotency_key: str = Field(..., min_length=1, description="Unique idempotency token")

    @field_validator("content_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not re.match(r"^[a-fA-F0-9]{64}$", value):
            raise ValueError("content_sha256 must be a valid 64-character hex string")
        return value.lower()


class IngestDispatchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    success: bool = True
    external_task_id: str
    status: DispatchStatus
    message: str
    data: dict[str, Any] | None = None


class IngestCallbackPayload(BaseModel):
    """Signed AI-service -> backend callback body (design B.3).

    It describes one processing event. It carries no tenant, user, role or
    destination: the tenant travels only in the signed system context, and the
    destination is BACKEND_CALLBACK_URL. Unknown fields are rejected.
    """

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    event_id: str = Field(
        ...,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        description="UUIDv4 chosen by AI-service, unique per event",
    )
    job_id: str = Field(..., min_length=1)
    external_task_id: str = Field(..., min_length=1)
    run_number: int = Field(..., ge=1)
    attempt_count: int = Field(..., ge=1)
    event_type: Literal["stage_progress", "job_completed", "job_failed"]
    stage: Literal["extracted", "chunked", "indexed"] | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    is_retryable: bool = False
    # Evidence of what happened before a failure (the backend stores it as
    # the job's error_details), e.g. a verified fetch. Data only.
    details: dict[str, Any] = Field(default_factory=dict)


class IngestTaskStatusResponse(BaseModel):
    """Signed status of one ingestion task, for backend reconciliation
    (M3-09). ``events`` are the exact callback bodies the worker produced,
    in order; the backend applies them through its callback rules."""

    external_task_id: str
    job_id: str
    run_number: int
    attempt_count: int
    state: Literal["accepted", "running", "succeeded", "failed"]
    events: list[IngestCallbackPayload]
