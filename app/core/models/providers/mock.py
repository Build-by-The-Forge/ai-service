import asyncio
from collections.abc import AsyncIterator

from app.core.models.contracts import (
    ModelDecision,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelStreamChunk,
    ModelUsage,
)
from app.core.models.provider import ModelProviderAdapter


class MockProviderAdapter(ModelProviderAdapter):
    """
    Deterministic mock provider for unit testing gateway policies, resilience, and streaming.
    Supports dynamic response configuration and fault injection.
    """

    def __init__(
        self,
        name: str = "mock",
        default_response: str = "Mock generated content",
        latency_seconds: float = 0.0,
        simulated_error: Exception | None = None,
        token_usage: ModelUsage | None = None,
    ):
        self._name = name
        self._default_response = default_response
        self._latency = latency_seconds
        self._simulated_error = simulated_error
        self._token_usage = token_usage or ModelUsage(
            prompt_tokens=10, completion_tokens=20, total_tokens=30, estimated_cost_usd=0.00005
        )
        self.call_history: list[ModelRequest] = []

    @property
    def name(self) -> str:
        return self._name

    def supports_model(self, model: str) -> bool:
        return True

    def set_simulated_error(self, exc: Exception | None) -> None:
        self._simulated_error = exc

    def set_default_response(self, text: str) -> None:
        self._default_response = text

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        self.call_history.append(request)
        if self._latency > 0:
            await asyncio.sleep(self._latency)

        if self._simulated_error:
            raise self._simulated_error

        return ModelResponse(
            content=self._default_response,
            model=decision.model,
            provider=self._name,
            role=ModelRole.ASSISTANT,
            finish_reason="stop",
            usage=self._token_usage,
            latency_ms=round(self._latency * 1000.0, 2),
            context=request.context,
        )

    async def generate_stream(
        self, request: ModelRequest, decision: ModelDecision
    ) -> AsyncIterator[ModelStreamChunk]:
        self.call_history.append(request)
        if self._latency > 0:
            await asyncio.sleep(self._latency)

        if self._simulated_error:
            raise self._simulated_error

        words = self._default_response.split(" ")
        for idx, word in enumerate(words):
            chunk_text = word if idx == 0 else f" {word}"
            yield ModelStreamChunk(
                delta_content=chunk_text,
                model=decision.model,
                provider=self._name,
                finish_reason="stop" if idx == len(words) - 1 else None,
            )
            await asyncio.sleep(0.001)
