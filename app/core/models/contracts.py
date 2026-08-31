import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ModelMessage:
    role: ModelRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": self.role.value if isinstance(self.role, ModelRole) else str(self.role),
            "content": self.content,
        }
        if self.name:
            data["name"] = self.name
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        return data


@dataclass(frozen=True)
class ModelInvocationContext:
    """
    Explicit execution metadata passed into Model Gateway.
    Decoupled from AIExecutionContext internals.
    """

    tenant_id: str
    execution_id: str
    task_id: str
    correlation_id: str
    agent_run_id: str | None = None
    model_call_id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ModelPolicy:
    """
    High-level declarative policy for model execution.
    Callers express goals and constraints without hardcoding provider details.
    """

    quality: str = "balanced"  # "fast", "balanced", "high"
    preferred_provider: str = "openai"
    fallback_providers: list[str] = field(default_factory=list)
    max_tokens: int = 2048
    max_cost_usd: float | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 3
    temperature: float = 0.7


@dataclass(frozen=True)
class ModelDecision:
    """
    Immutable execution decision resolved from ModelPolicy before provider dispatch.
    """

    provider: str
    model: str
    fallback_providers: list[str]
    max_tokens: int
    timeout_seconds: float
    max_retries: int
    max_cost_usd: float | None
    temperature: float


@dataclass(frozen=True)
class ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def __add__(self, other: "ModelUsage") -> "ModelUsage":
        return ModelUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost_usd=self.estimated_cost_usd + other.estimated_cost_usd,
        )


@dataclass(frozen=True)
class ModelRequest:
    """
    Provider-independent request contract for all model invocations.
    """

    messages: list[ModelMessage]
    context: ModelInvocationContext
    policy: ModelPolicy | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    response_format: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelResponse:
    """
    Normalized response returned by ModelGateway regardless of underlying provider.
    """

    content: str
    model: str
    provider: str
    role: ModelRole = ModelRole.ASSISTANT
    finish_reason: str | None = "stop"
    tool_calls: list[dict[str, Any]] | None = None
    usage: ModelUsage = field(default_factory=ModelUsage)
    latency_ms: float = 0.0
    context: ModelInvocationContext | None = None


@dataclass(frozen=True)
class ModelStreamChunk:
    """
    Incremental chunk yielded during streaming model generation.
    """

    delta_content: str
    model: str
    provider: str
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    usage: ModelUsage | None = None
