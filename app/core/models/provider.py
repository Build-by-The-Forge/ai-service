import abc
from collections.abc import AsyncIterator

from app.core.models.contracts import (
    ModelDecision,
    ModelRequest,
    ModelResponse,
    ModelStreamChunk,
)


class ModelProviderAdapter(abc.ABC):
    """
    Abstract interface for concrete LLM provider adapters (OpenAI, Gemini, Anthropic, Mock).
    Translates ModelRequest into provider SDK calls and normalizes responses to ModelResponse.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the provider (e.g. 'openai', 'gemini', 'mock')."""

    @abc.abstractmethod
    def supports_model(self, model: str) -> bool:
        """Returns True if this provider can execute the given model name."""

    @abc.abstractmethod
    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        """Executes a non-streaming completion and returns a normalized ModelResponse."""

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """The most a call could cost, reserved against a cost budget before it
        is made. Adapters with pricing override it; 0.0 reserves no cost."""
        return 0.0

    @abc.abstractmethod
    def generate_stream(
        self, request: ModelRequest, decision: ModelDecision
    ) -> AsyncIterator[ModelStreamChunk]:
        """Executes a streaming completion and yields ModelStreamChunk instances."""
