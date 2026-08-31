import re
from enum import Enum
from typing import Any, Dict, Optional
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
    model_config = ConfigDict(extra="ignore", from_attributes=True)

    job_id: str = Field(..., min_length=1, description="Backend Ingestion Job identifier")
    asset_id: str = Field(..., min_length=1, description="Content Asset identifier")
    tenant_id: str = Field(..., min_length=1, description="Tenant/School identifier")
    source: str = Field(default="upload", description="Source of the asset (e.g. upload, s3, url)")
    authority: ContentAuthority = Field(
        default=ContentAuthority.INSTITUTIONAL, description="Content authority tier"
    )
    storage_uri: str = Field(..., min_length=1, description="URI or path where raw asset is stored")
    mime_type: str = Field(..., min_length=1, description="MIME type of the asset")
    content_sha256: str = Field(
        ..., min_length=64, max_length=64, description="Hex-encoded SHA-256 hash of content"
    )
    run_number: int = Field(default=1, ge=1, description="Execution run number")
    attempt_count: int = Field(default=1, ge=1, description="Execution attempt count")
    correlation_id: str = Field(..., min_length=1, description="Distributed tracing correlation ID")
    idempotency_key: str = Field(..., min_length=1, description="Unique idempotency token")
    callback_url: str = Field(
        ..., min_length=1, description="Webhook callback URL for progress/results"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary course/topic/asset metadata"
    )

    @field_validator("content_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not re.match(r"^[a-fA-F0-9]{64}$", value):
            raise ValueError("content_sha256 must be a valid 64-character hex string")
        return value.lower()

    @field_validator("callback_url")
    @classmethod
    def validate_callback_url(cls, value: str) -> str:
        if not (value.startswith("http://") or value.startswith("https://")):
            raise ValueError("callback_url must start with http:// or https://")
        return value


class IngestDispatchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    success: bool = True
    external_task_id: str
    status: DispatchStatus
    message: str
    data: Optional[Dict[str, Any]] = None


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
    progress_pct: Optional[int] = Field(default=None, ge=0, le=100)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retryable: Optional[bool] = None
    extracted_metadata: Optional[Dict[str, Any]] = None
    chunk_summary: Optional[Dict[str, Any]] = None
