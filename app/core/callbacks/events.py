import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CallbackEventType(str, Enum):
    STARTED = "STARTED"
    PROGRESS = "PROGRESS"
    STAGE_COMPLETED = "STAGE_COMPLETED"  # a real pipeline stage finished
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class CallbackEvent:
    """
    Immutable domain event emitted during task execution to report lifecycle facts to external callers.
    Carries full execution and tenant context.
    """

    job_id: str
    asset_id: str
    tenant_id: str
    run_number: int
    attempt_count: int
    external_task_id: str
    correlation_id: str
    event_type: CallbackEventType
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    progress_pct: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool | None = None
    extracted_metadata: dict[str, Any] | None = None
    chunk_summary: dict[str, Any] | None = None
    stage: str | None = None
    metrics: dict[str, Any] | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "job_id": self.job_id,
            "asset_id": self.asset_id,
            "tenant_id": self.tenant_id,
            "run_number": self.run_number,
            "attempt_count": self.attempt_count,
            "external_task_id": self.external_task_id,
            "correlation_id": self.correlation_id,
            "event_type": (
                self.event_type.value
                if isinstance(self.event_type, CallbackEventType)
                else str(self.event_type)
            ),
            "event_id": self.event_id,
            "timestamp": self.timestamp,
        }
        if self.progress_pct is not None:
            data["progress_pct"] = self.progress_pct
        if self.error_code is not None:
            data["error_code"] = self.error_code
        if self.error_message is not None:
            data["error_message"] = self.error_message
        if self.retryable is not None:
            data["retryable"] = self.retryable
        if self.extracted_metadata is not None:
            data["extracted_metadata"] = self.extracted_metadata
        if self.chunk_summary is not None:
            data["chunk_summary"] = self.chunk_summary
        if self.stage is not None:
            data["stage"] = self.stage
        if self.metrics is not None:
            data["metrics"] = self.metrics
        return data


@dataclass(frozen=True)
class CallbackDeliveryResult:
    """
    Result of a callback delivery attempt by the transport dispatcher.
    Decoupled from execution state.
    """

    success: bool
    status_code: int | None = None
    attempts: int = 1
    retryable: bool = False
    error_code: str | None = None
    error_message: str | None = None
    delivered_at: float | None = None
    event_id: str | None = None
