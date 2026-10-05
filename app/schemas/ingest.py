import re
from enum import Enum
from typing import Any

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


class CallbackEventType(str, Enum):
    STARTED = "STARTED"
    PROGRESS = "PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


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
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    job_id: str
    asset_id: str
    tenant_id: str
    run_number: int
    attempt_count: int
    external_task_id: str
    correlation_id: str
    event_type: CallbackEventType
    progress_pct: int | None = Field(default=None, ge=0, le=100)
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool | None = None
    extracted_metadata: dict[str, Any] | None = None
    chunk_summary: dict[str, Any] | None = None
