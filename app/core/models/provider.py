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

    @abc.abstractmethod
    async def generate_stream(
        self, request: ModelRequest, decision: ModelDecision
    ) -> AsyncIterator[ModelStreamChunk]:
        """Executes a streaming completion and yields ModelStreamChunk instances."""
