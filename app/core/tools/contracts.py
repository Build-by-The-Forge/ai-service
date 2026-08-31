from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional
import uuid


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
    agent_run_id: Optional[str] = None
    actor_role: str = "assistant"
    permissions: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    deadline: Optional[float] = None
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

    allowed_tools: Optional[List[str]] = None  # None means all authorized tools in registry
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
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None
    required_permissions: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    tenant_scoped: bool = True
    side_effect: ToolSideEffect = ToolSideEffect.READ_ONLY
    timeout_seconds: float = 15.0
    max_invocations: Optional[int] = None
    is_enabled: bool = True

    def to_openai_tool(self) -> Dict[str, Any]:
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
    arguments: Dict[str, Any]
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
    data: Optional[Any] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: float = 0.0
    output_bytes: int = 0
