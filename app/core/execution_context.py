import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    DEAD_LETTER = "DEAD_LETTER"


@dataclass(frozen=True)
class AITask:
    """
    Immutable typed task envelope that travels through transports and queues.
    Carries the full security and execution identity.
    """

    task_id: str
    task_type: str
    execution_id: str
    tenant_id: str
    job_id: str
    asset_id: str
    run_number: int
    attempt_count: int
    correlation_id: str
    payload: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    deadline: float | None = None
    max_attempts: int = 3

    def is_expired(self) -> bool:
        if self.deadline is None:
            return False
        return time.time() > self.deadline

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "execution_id": self.execution_id,
            "tenant_id": self.tenant_id,
            "job_id": self.job_id,
            "asset_id": self.asset_id,
            "run_number": self.run_number,
            "attempt_count": self.attempt_count,
            "correlation_id": self.correlation_id,
            "payload": self.payload,
            "created_at": self.created_at,
            "deadline": self.deadline,
            "max_attempts": self.max_attempts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AITask":
        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            execution_id=data["execution_id"],
            tenant_id=data["tenant_id"],
            job_id=data["job_id"],
            asset_id=data["asset_id"],
            run_number=int(data.get("run_number", 1)),
            attempt_count=int(data.get("attempt_count", 1)),
            correlation_id=data["correlation_id"],
            payload=data.get("payload", {}),
            created_at=float(data.get("created_at", time.time())),
            deadline=float(data["deadline"]) if data.get("deadline") is not None else None,
            max_attempts=int(data.get("max_attempts", 3)),
        )


class AIExecutionContext:
    """
    Mutable runtime execution context established by the TaskWorker for a single task run.
    Contains authorization boundaries, cooperative cancellation tokens, and runtime telemetry.
    """

    def __init__(
        self,
        execution_id: str,
        tenant_id: str,
        job_id: str,
        asset_id: str,
        run_number: int,
        attempt_count: int,
        correlation_id: str,
        task_type: str,
        actor_context: dict[str, Any] | None = None,
        permissions: list[str] | None = None,
        created_at: float | None = None,
        deadline: float | None = None,
    ):
        self.execution_id = execution_id
        self.tenant_id = tenant_id
        self.job_id = job_id
        self.asset_id = asset_id
        self.run_number = run_number
        self.attempt_count = attempt_count
        self.correlation_id = correlation_id
        self.task_type = task_type
        self.actor_context = actor_context or {}
        self.permissions = permissions or []
        self.created_at = created_at or time.time()
        self.deadline = deadline
        self._is_cancelled = False
        self._cancellation_reason: str | None = None
        self._metadata: dict[str, Any] = {}

    @classmethod
    def from_task(
        cls,
        task: AITask,
        actor_context: dict[str, Any] | None = None,
        permissions: list[str] | None = None,
    ) -> "AIExecutionContext":
        return cls(
            execution_id=task.execution_id or f"exec_{uuid.uuid4().hex}",
            tenant_id=task.tenant_id,
            job_id=task.job_id,
            asset_id=task.asset_id,
            run_number=task.run_number,
            attempt_count=task.attempt_count,
            correlation_id=task.correlation_id,
            task_type=task.task_type,
            actor_context=actor_context,
            permissions=permissions,
            created_at=task.created_at,
            deadline=task.deadline,
        )

    def cancel(self, reason: str = "User or system requested cancellation") -> None:
        self._is_cancelled = True
        self._cancellation_reason = reason

    def is_cancelled(self) -> bool:
        return self._is_cancelled

    @property
    def cancellation_reason(self) -> str | None:
        return self._cancellation_reason

    def is_expired(self) -> bool:
        if self.deadline is None:
            return False
        return time.time() > self.deadline

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)
