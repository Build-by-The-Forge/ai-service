from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, Optional
import uuid


class AgentEventType(str, Enum):
    AGENT_STARTED = "AGENT_STARTED"
    AGENT_STEP_STARTED = "AGENT_STEP_STARTED"
    MODEL_CALL_STARTED = "MODEL_CALL_STARTED"
    MODEL_CALL_COMPLETED = "MODEL_CALL_COMPLETED"
    TOOL_CALL_STARTED = "TOOL_CALL_STARTED"
    TOOL_CALL_COMPLETED = "TOOL_CALL_COMPLETED"
    AGENT_STEP_COMPLETED = "AGENT_STEP_COMPLETED"
    AGENT_COMPLETED = "AGENT_COMPLETED"
    AGENT_FAILED = "AGENT_FAILED"
    AGENT_CANCELLED = "AGENT_CANCELLED"
    AGENT_BUDGET_EXCEEDED = "AGENT_BUDGET_EXCEEDED"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"


@dataclass(frozen=True)
class AgentEvent:
    """
    Structured observability event emitted at every stage of the agent reasoning loop.
    """

    event_type: AgentEventType
    run_id: str
    agent_id: str
    tenant_id: str
    execution_id: str
    task_id: str
    correlation_id: str
    step_number: int = 0
    data: Optional[Dict[str, Any]] = None
    event_id: str = field(default_factory=lambda: f"aevt_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "execution_id": self.execution_id,
            "task_id": self.task_id,
            "correlation_id": self.correlation_id,
            "step_number": self.step_number,
            "timestamp": self.timestamp,
        }
        if self.data:
            result["data"] = self.data
        return result
