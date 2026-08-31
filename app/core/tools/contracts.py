import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolSideEffect(str, Enum):
    READ_ONLY = "READ_ONLY"
    ANALYTICAL = "ANALYTICAL"
    PROPOSAL = "PROPOSAL"
    MUTATING = "MUTATING"


@dataclass(frozen=True)
class ToolContext:
    """
    Authoritative caller security and execution context passed into the tool execution plane.
    Cannot be modified or escalated by tool arguments or LLM outputs.
    """

    tenant_id: str
    execution_id: str
    task_id: str
    correlation_id: str
    agent_run_id: str | None = None
    actor_role: str = "assistant"
    permissions: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    deadline: float | None = None
    _is_cancelled: bool = False

    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def is_expired(self) -> bool:
        if self.deadline is None:
            return False
        return time.time() > self.deadline


@dataclass(frozen=True)
class ToolExecutionPolicy:
    """
    Execution constraints governing tool capabilities during an execution run.
    """

    allowed_tools: list[str] | None = None  # None means all authorized tools in registry
    max_tool_calls: int = 10
    max_runtime_seconds: float = 60.0
    max_input_bytes: int = 65536  # 64 KB
    max_output_bytes: int = 262144  # 256 KB
    allow_proposals: bool = True
    allow_mutating: bool = False  # Strictly False by default in AI Service


@dataclass(frozen=True)
class ToolDefinition:
    """
    Metadata specification and capability boundaries of a registered tool.
    """

    name: str
    version: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    required_permissions: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    tenant_scoped: bool = True
    side_effect: ToolSideEffect = ToolSideEffect.READ_ONLY
    timeout_seconds: float = 15.0
    max_invocations: int | None = None
    is_enabled: bool = True

    def to_openai_tool(self) -> dict[str, Any]:
        """Converts ToolDefinition to OpenAI-compatible function tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(frozen=True)
class ToolInvocation:
    """
    Typed invocation envelope requested by caller / model.
    """

    tool_name: str
    arguments: dict[str, Any]
    context: ToolContext
    invocation_id: str = field(default_factory=lambda: f"tcall_{uuid.uuid4().hex[:12]}")
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ToolResult:
    """
    Normalized result of a tool execution. Decoupled from caller runtime state.
    """

    invocation_id: str
    tool_name: str
    success: bool
    data: Any | None = None
    error_code: str | None = None
    error_message: str | None = None
    latency_ms: float = 0.0
    output_bytes: int = 0
