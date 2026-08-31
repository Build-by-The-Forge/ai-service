import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from app.core.models.contracts import ModelPolicy, ModelUsage
from app.core.tools.contracts import ToolExecutionPolicy


class AgentState(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING_FOR_TOOL = "WAITING_FOR_TOOL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    MAX_STEPS_EXCEEDED = "MAX_STEPS_EXCEEDED"


@dataclass(frozen=True)
class AgentContext:
    """
    Authoritative caller security and execution context passed into the Agent Runtime.
    Cannot be modified or overridden by LLM reasoning steps or tool outputs.
    """

    tenant_id: str
    execution_id: str
    task_id: str
    correlation_id: str
    agent_run_id: str = field(default_factory=lambda: f"arun_{uuid.uuid4().hex[:12]}")
    actor_role: str = "assistant"
    permissions: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    deadline: float | None = None
    _is_cancelled: bool = False
    _cancellation_reason: str | None = None

    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def cancellation_reason(self) -> str | None:
        return self._cancellation_reason

    def is_expired(self) -> bool:
        if self.deadline is None:
            return False
        return time.time() > self.deadline


@dataclass(frozen=True)
class AgentExecutionPolicy:
    """
    Hard runtime boundaries governing multi-step reasoning.
    """

    max_steps: int = 10
    max_model_calls: int = 10
    max_tool_calls: int = 10
    max_tokens: int = 8192
    max_cost_usd: float | None = None
    max_runtime_seconds: float = 60.0
    max_consecutive_failures: int = 3
    max_output_bytes: int = 262144  # 256 KB


@dataclass(frozen=True)
class AgentDefinition:
    """
    Immutable specification of an agent's capability bounds, system prompt, and policies.
    """

    agent_id: str
    name: str
    version: str
    description: str
    system_prompt: str
    model_policy: ModelPolicy = field(default_factory=ModelPolicy)
    tool_policy: ToolExecutionPolicy = field(default_factory=ToolExecutionPolicy)
    execution_policy: AgentExecutionPolicy = field(default_factory=AgentExecutionPolicy)
    allowed_tools: list[str] | None = None
    required_permissions: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    allow_proposals: bool = True
    allow_mutating: bool = False
    is_enabled: bool = True


@dataclass
class AgentRun:
    """
    Mutable state tracker maintaining monotonic lifecycle invariants during an agent execution.
    """

    run_id: str
    agent_id: str
    agent_version: str
    tenant_id: str
    execution_id: str
    task_id: str
    correlation_id: str
    state: AgentState = AgentState.CREATED
    step_count: int = 0
    model_call_count: int = 0
    tool_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    deadline: float | None = None
    last_error: str | None = None
    consecutive_failures: int = 0

    def is_active(self) -> bool:
        return self.state in (AgentState.CREATED, AgentState.RUNNING, AgentState.WAITING_FOR_TOOL)

    def record_model_call(self, usage: ModelUsage) -> None:
        self.model_call_count += 1
        self.input_tokens += usage.prompt_tokens
        self.output_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.estimated_cost_usd += usage.estimated_cost_usd

    def record_tool_call(self) -> None:
        self.tool_call_count += 1

    def record_step(self) -> None:
        self.step_count += 1

    def transition_to(self, new_state: AgentState, error: str | None = None) -> None:
        if not self.is_active() and new_state != self.state:
            raise ValueError(f"Cannot transition terminal run from {self.state} to {new_state}")
        self.state = new_state
        if error:
            self.last_error = error
        if new_state in (
            AgentState.COMPLETED,
            AgentState.FAILED,
            AgentState.CANCELLED,
            AgentState.TIMED_OUT,
            AgentState.BUDGET_EXCEEDED,
            AgentState.MAX_STEPS_EXCEEDED,
        ):
            self.completed_at = time.time()


@dataclass(frozen=True)
class AgentResult:
    """
    Deterministic machine-readable outcome returned to the caller upon agent completion.
    """

    run_id: str
    agent_id: str
    state: AgentState
    output: str | None = None
    step_count: int = 0
    model_call_count: int = 0
    tool_call_count: int = 0
    usage: ModelUsage = field(default_factory=ModelUsage)
    estimated_cost_usd: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    latency_ms: float = 0.0
    error_code: str | None = None
    error_message: str | None = None
