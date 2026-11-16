"""Gemini through the model gateway, with the google-genai SDK.

Only the gateway calls this adapter, and only when "gemini" is both
registered and approved (MODEL_PROVIDERS_APPROVED, G-33). It is the one
module in app/ that may import the SDK (tests/unit/test_model_interface.py).
The SDK is configured so the gateway keeps control of every call:

- **Key:** passed explicitly, with vertexai=False, so the SDK's own
  environment variables (GOOGLE_API_KEY, GOOGLE_GENAI_USE_VERTEXAI...) are
  never read. Without GEMINI_API_KEY nothing is sent.
- **Retries:** the SDK's are off (one attempt). The gateway owns retries,
  the request's budget and its one deadline.
- **Timeout:** the attempt's, which the gateway caps at the time left.
- **Errors:** by status code only. The SDK's error text quotes the
  provider's response body, so it is never kept, logged or raised.

Data use: on the free tier and the "Contributor" tier, Google may use the
content to improve its products; on the paid Standard tier it doesn't
(Gemini API pricing page, checked 2026-11-15). Free-tier keys are for
development with synthetic material only. Student or institutional material
goes only to a paid-tier key, and only once G-33 approves the provider.
"""

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

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
    ModelGatewayError,
    ModelTimeoutError,
    ProviderUnavailableError,
    RateLimitExceededError,
)
from app.core.models.provider import ModelProviderAdapter

NAME = "gemini"

# USD per 1,000 tokens: (input, output including thinking). The paid Standard
# tier's 2027 prices ($1.50 and $7.50 per 1M); until 2026-12-31 they are half
# that. The gateway reserves an estimate before each call, so it estimates
# high rather than low. An unlisted model is estimated at the highest listed.
GEMINI_PRICING: dict[str, tuple[float, float]] = {
    "gemini-3.8-flash": (0.0015, 0.0075),
}


def calculate_gemini_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = GEMINI_PRICING.get(model) or max(GEMINI_PRICING.values(), key=sum)
    return round(prompt_tokens / 1000.0 * rates[0] + completion_tokens / 1000.0 * rates[1], 6)


def _no_answer() -> ModelGatewayError:
    """No user-visible text came back. Asking again is unlikely to change it."""
    return ModelGatewayError("Gemini returned no answer", provider=NAME, retryable=False)


class GeminiProviderAdapter(ModelProviderAdapter):
    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        # No placeholder: without a key, nothing is sent (_require_key).
        self._api_key = api_key or settings.GEMINI_API_KEY
        # Tests pass a client over httpx.MockTransport: the real SDK, no network.
        self._http_client = http_client

    @property
    def name(self) -> str:
        return NAME

    def supports_model(self, model: str) -> bool:
        return model.startswith("gemini-")

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        return calculate_gemini_cost(model, prompt_tokens, completion_tokens)

    def _require_key(self) -> str:
        if not self._api_key:
            raise AuthenticationError("Gemini API key is not configured", provider=NAME)
        return self._api_key

    def _client(self, decision: ModelDecision) -> genai.Client:
        return genai.Client(
            vertexai=False,
            api_key=self._require_key(),
            http_options=types.HttpOptions(
                timeout=max(1, int(decision.timeout_seconds * 1000)),
                retry_options=types.HttpRetryOptions(attempts=1),
                httpx_async_client=self._http_client,
            ),
        )

    def _contents(
        self, request: ModelRequest, decision: ModelDecision
    ) -> tuple[list[types.Content], types.GenerateContentConfig]:
        if request.tools or request.tool_choice:
            raise InvalidRequestError(
                "Tool calls aren't supported through the Gemini adapter", provider=NAME
            )
        system: list[str] = []
        contents: list[types.Content] = []
        for message in request.messages:
            if message.role == ModelRole.SYSTEM:
                system.append(message.content or "")
            elif message.role in (ModelRole.USER, ModelRole.ASSISTANT):
                role = "user" if message.role == ModelRole.USER else "model"
                contents.append(
                    types.Content(role=role, parts=[types.Part(text=message.content or "")])
                )
            else:
                raise InvalidRequestError(
                    "Only system, user and assistant messages go to Gemini", provider=NAME
                )
        json_mode = request.structured is not None or (
            isinstance(request.response_format, dict)
            and request.response_format.get("type") == "json_object"
        )
        config = types.GenerateContentConfig(
            system_instruction="\n\n".join(system) if system else None,
            temperature=decision.temperature,
            max_output_tokens=decision.max_tokens,
            top_p=request.top_p,
            # JSON mode. The kind and hints are for the offline adapter: the
            # prompt says what the object holds, and they aren't sent.
            response_mime_type="application/json" if json_mode else None,
        )
        return contents, config

    @staticmethod
    def _failure(exc: Exception) -> ModelGatewayError:
        """The gateway's error for a failed call, built from the status code or
        the exception's type only. The SDK's message quotes the provider's
        response body, which isn't ours to keep, and can quote the request.
        Callers raise it outside their except block, so the original isn't
        even attached as its context."""
        if isinstance(exc, httpx.TimeoutException):
            return ModelTimeoutError("Gemini request timed out", provider=NAME)
        if isinstance(exc, httpx.NetworkError):
            return ProviderUnavailableError(
                f"Gemini network error ({type(exc).__name__})", provider=NAME
            )
        status = (
            exc.code if isinstance(exc, genai_errors.APIError) and isinstance(exc.code, int) else 0
        )
        message = f"Gemini returned HTTP {status}"
        mapped: ModelGatewayError
        if status in (401, 403):
            mapped = AuthenticationError(message, provider=NAME)
        elif status == 429:
            mapped = RateLimitExceededError(message, provider=NAME)
        elif status in (400, 404, 422):
            mapped = InvalidRequestError(message, provider=NAME)
        elif status >= 500:
            mapped = ProviderUnavailableError(message, provider=NAME)
        else:
            mapped = ModelGatewayError(message, provider=NAME, retryable=False)
        mapped.status_code = status
        return mapped

    @staticmethod
    def _check_prompt(response: Any) -> None:
        """A blocked prompt is a non-retryable failure, with no provider text.
        Gemini reports it at the prompt level, usually with no candidates."""
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            raise ModelGatewayError("Gemini blocked the prompt", provider=NAME, retryable=False)

    @staticmethod
    def _answer(response: Any) -> tuple[str, str | None]:
        """The answer's text (no thought parts) and its finish reason, None while
        a stream continues. A blocked or withheld answer is a non-retryable
        failure, with no provider text."""
        GeminiProviderAdapter._check_prompt(response)
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise _no_answer()
        candidate = candidates[0]
        reason = getattr(candidate, "finish_reason", None)
        reason_name = (getattr(reason, "name", None) or str(reason)).lower() if reason else None
        if reason_name in ("finish_reason_unspecified",):
            reason_name = None
        if reason_name in ("safety", "blocklist", "prohibited_content", "spii", "recitation"):
            raise ModelGatewayError("Gemini withheld the answer", provider=NAME, retryable=False)
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        text = "".join(
            p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)
        )
        return text, reason_name

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        start = time.time()
        contents, config = self._contents(request, decision)
        client = self._client(decision)
        failure: ModelGatewayError | None = None
        try:
            response = await client.aio.models.generate_content(
                model=decision.model, contents=contents, config=config
            )
        except (genai_errors.APIError, httpx.TimeoutException, httpx.NetworkError) as exc:
            failure = self._failure(exc)
        finally:
            if self._http_client is None:
                await client.aio.aclose()
        if failure is not None:
            raise failure

        text, finish_reason = self._answer(response)
        if not text.strip():
            # No visible text once thought parts are removed: an empty or
            # thought-only candidate is no answer, not an empty success.
            raise _no_answer()
        usage_meta = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage_meta, "prompt_token_count", None) or 0
        # Thinking tokens are billed as output.
        completion_tokens = (getattr(usage_meta, "candidates_token_count", None) or 0) + (
            getattr(usage_meta, "thoughts_token_count", None) or 0
        )
        total_tokens = (
            getattr(usage_meta, "total_token_count", None) or prompt_tokens + completion_tokens
        )
        return ModelResponse(
            content=text,
            model=decision.model,
            provider=NAME,
            role=ModelRole.ASSISTANT,
            finish_reason=finish_reason or "stop",
            usage=ModelUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=calculate_gemini_cost(
                    decision.model, prompt_tokens, completion_tokens
                ),
            ),
            latency_ms=round((time.time() - start) * 1000.0, 2),
            context=request.context,
        )

    async def generate_stream(
        self, request: ModelRequest, decision: ModelDecision
    ) -> AsyncIterator[ModelStreamChunk]:
        contents, config = self._contents(request, decision)
        client = self._client(decision)
        failure: ModelGatewayError | None = None
        visible = False  # any user-visible text so far
        try:
            stream = await client.aio.models.generate_content_stream(
                model=decision.model, contents=contents, config=config
            )
            async for response in stream:
                if not (getattr(response, "candidates", None) or []):
                    # A chunk without candidates (usage only, say) carries no
                    # text, but it can carry the prompt's block: check first.
                    self._check_prompt(response)
                    continue
                text, finish_reason = self._answer(response)
                visible = visible or bool(text.strip())
                if finish_reason is None and not text:
                    continue  # thought-only while thinking: nothing to deliver
                if finish_reason is not None and not visible:
                    # Finished with no visible text: no answer, never an
                    # empty success, whatever the finish reason says.
                    raise _no_answer()
                yield ModelStreamChunk(
                    delta_content=text,
                    model=decision.model,
                    provider=NAME,
                    finish_reason=finish_reason,
                )
        except (genai_errors.APIError, httpx.TimeoutException, httpx.NetworkError) as exc:
            failure = self._failure(exc)
        finally:
            if self._http_client is None:
                await client.aio.aclose()
        if failure is not None:
            raise failure
        if not visible:
            raise _no_answer()  # the stream ended without any visible text
