import time
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from app.core.config import settings
from app.core.models.contracts import (
    ModelDecision,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelStreamChunk,
    ModelUsage,
)
from app.core.models.errors import (
    AuthenticationError,
    InvalidRequestError,
    ModelTimeoutError,
    ProviderUnavailableError,
    RateLimitExceededError,
)
from app.core.models.provider import ModelProviderAdapter

# Standard pricing table: (input_price_per_1k, output_price_per_1k)
OPENAI_PRICING: Dict[str, tuple[float, float]] = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-3.5-turbo": (0.0015, 0.002),
}


def calculate_openai_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    # Default to gpt-4o-mini pricing if unrecognized
    rates = OPENAI_PRICING.get(model, (0.001, 0.002))
    cost = (prompt_tokens / 1000.0 * rates[0]) + (completion_tokens / 1000.0 * rates[1])
    return round(cost, 6)


class OpenAIProviderAdapter(ModelProviderAdapter):
    """
    Production OpenAI adapter conforming to ModelProviderAdapter.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self._api_key = api_key or settings.OPENAI_API_KEY or "sk-mock-key"
        self._base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

    @property
    def name(self) -> str:
        return "openai"

    def supports_model(self, model: str) -> bool:
        return (
            model.startswith("gpt-")
            or model.startswith("o1-")
            or model.startswith("o3-")
            or "openai" in model
        )

    def _prepare_payload(
        self, request: ModelRequest, decision: ModelDecision, stream: bool = False
    ) -> Dict[str, Any]:
        messages_payload = [msg.to_dict() for msg in request.messages]
        payload: Dict[str, Any] = {
            "model": decision.model,
            "messages": messages_payload,
            "temperature": decision.temperature,
            "max_tokens": decision.max_tokens,
            "stream": stream,
        }
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.tools:
            payload["tools"] = request.tools
        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice
        if request.response_format:
            payload["response_format"] = request.response_format
        return payload

    def _map_http_error(self, status_code: int, response_text: str) -> None:
        if status_code in (401, 403):
            raise AuthenticationError(
                f"OpenAI authentication failed: {response_text}", provider="openai"
            )
        elif status_code == 429:
            raise RateLimitExceededError(
                f"OpenAI rate limit exceeded: {response_text}", provider="openai"
            )
        elif status_code in (400, 422):
            raise InvalidRequestError(f"OpenAI invalid request: {response_text}", provider="openai")
        elif status_code >= 500:
            raise ProviderUnavailableError(
                f"OpenAI provider error ({status_code}): {response_text}", provider="openai"
            )
        else:
            raise ProviderUnavailableError(
                f"OpenAI request failed ({status_code}): {response_text}", provider="openai"
            )

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        start_time = time.time()
        payload = self._prepare_payload(request, decision, stream=False)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=decision.timeout_seconds) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code != 200:
                    self._map_http_error(resp.status_code, resp.text)

                data = resp.json()
                latency_ms = (time.time() - start_time) * 1000.0

                choice = data["choices"][0]
                message = choice.get("message", {})
                content = message.get("content") or ""
                role_str = message.get("role", "assistant")
                tool_calls = message.get("tool_calls")
                finish_reason = choice.get("finish_reason", "stop")

                raw_usage = data.get("usage", {})
                prompt_tokens = raw_usage.get("prompt_tokens", 0)
                completion_tokens = raw_usage.get("completion_tokens", 0)
                total_tokens = raw_usage.get("total_tokens", prompt_tokens + completion_tokens)
                cost_usd = calculate_openai_cost(decision.model, prompt_tokens, completion_tokens)

                usage = ModelUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=cost_usd,
                )

                role_enum = (
                    ModelRole(role_str)
                    if role_str in ModelRole._value2member_map_
                    else ModelRole.ASSISTANT
                )

                return ModelResponse(
                    content=content,
                    model=decision.model,
                    provider="openai",
                    role=role_enum,
                    finish_reason=finish_reason,
                    tool_calls=tool_calls,
                    usage=usage,
                    latency_ms=round(latency_ms, 2),
                    context=request.context,
                )

        except httpx.TimeoutException as te:
            raise ModelTimeoutError(
                f"OpenAI request timed out after {decision.timeout_seconds}s: {str(te)}",
                provider="openai",
            )
        except httpx.NetworkError as ne:
            raise ProviderUnavailableError(f"OpenAI network error: {str(ne)}", provider="openai")

    async def generate_stream(
        self, request: ModelRequest, decision: ModelDecision
    ) -> AsyncIterator[ModelStreamChunk]:
        payload = self._prepare_payload(request, decision, stream=True)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=decision.timeout_seconds) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        self._map_http_error(response.status_code, error_text.decode("utf-8"))

                    async for line in response.aiter_lines():
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            import json

                            chunk_data = json.loads(data_str)
                            choice = chunk_data["choices"][0]
                            delta = choice.get("delta", {})
                            delta_content = delta.get("content") or ""
                            finish_reason = choice.get("finish_reason")
                            tool_calls = delta.get("tool_calls")

                            yield ModelStreamChunk(
                                delta_content=delta_content,
                                model=decision.model,
                                provider="openai",
                                finish_reason=finish_reason,
                                tool_calls=tool_calls,
                            )

        except httpx.TimeoutException as te:
            raise ModelTimeoutError(f"OpenAI streaming timed out: {str(te)}", provider="openai")
        except httpx.NetworkError as ne:
            raise ProviderUnavailableError(
                f"OpenAI streaming network error: {str(ne)}", provider="openai"
            )
