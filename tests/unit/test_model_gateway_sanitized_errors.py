"""AI-01 PR 1, F2: provider text never reaches an exception or a log.

Errors are built from fixed text and the provider's status. What a provider
says back, and anything an adapter bug carries, may quote the prompt (a
student passage), so it isn't kept, at construction or in the logs.
"""

import logging
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.core.models.contracts import (
    ModelDecision,
    ModelInvocationContext,
    ModelMessage,
    ModelPolicy,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelStreamChunk,
)
from app.core.models.errors import (
    AuthenticationError,
    FallbackExhaustedError,
    InvalidRequestError,
    ModelGatewayError,
    ModelTimeoutError,
    ProviderUnavailableError,
    RateLimitExceededError,
)
from app.core.models.gateway import ModelGateway
from app.core.models.provider import ModelProviderAdapter
from app.core.models.providers.openai import OpenAIProviderAdapter

MARKER = "SECRET-STUDENT-PASSAGE"
CONTEXT = ModelInvocationContext(
    tenant_id="school_1", execution_id="exec_1", task_id="task_1", correlation_id="corr_1"
)
REQUEST = ModelRequest(
    messages=[ModelMessage(role=ModelRole.USER, content=MARKER)], context=CONTEXT
)
DECISION = ModelDecision(
    provider="openai",
    model="gpt-4o-mini",
    fallback_providers=[],
    max_tokens=100,
    timeout_seconds=10.0,
    max_retries=0,
    max_cost_usd=None,
    temperature=0.0,
)


def carries_no_marker(exc: BaseException) -> bool:
    texts = [str(exc), repr(exc), getattr(exc, "message", "")]
    return all(MARKER not in t for t in texts) and exc.__cause__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, error",
    [
        (400, InvalidRequestError),
        (401, AuthenticationError),
        (422, InvalidRequestError),
        (429, RateLimitExceededError),
        (503, ProviderUnavailableError),
        (404, ModelGatewayError),
    ],
)
async def test_a_provider_error_body_is_never_kept(status, error):
    adapter = OpenAIProviderAdapter(api_key="sk-test")
    body = f'{{"error": {{"message": "could not process: {MARKER}"}}}}'
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        post.return_value = httpx.Response(status_code=status, text=body)
        with pytest.raises(error) as caught:
            await adapter.generate(REQUEST, DECISION)

    assert carries_no_marker(caught.value)
    assert caught.value.status_code == status


@pytest.mark.asyncio
async def test_a_status_that_is_not_transient_is_not_retryable():
    adapter = OpenAIProviderAdapter(api_key="sk-test")
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        post.return_value = httpx.Response(status_code=404, text="no such route")
        with pytest.raises(ModelGatewayError) as caught:
            await adapter.generate(REQUEST, DECISION)

    assert caught.value.retryable is False


@pytest.mark.asyncio
async def test_a_malformed_success_body_is_not_kept():
    adapter = OpenAIProviderAdapter(api_key="sk-test")
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        post.return_value = httpx.Response(status_code=200, text=f"not json {MARKER}")
        with pytest.raises(ModelGatewayError) as caught:
            await adapter.generate(REQUEST, DECISION)

    assert carries_no_marker(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raised, error",
    [
        (httpx.ReadTimeout(MARKER), ModelTimeoutError),
        (httpx.ConnectError(MARKER), ProviderUnavailableError),
    ],
)
async def test_transport_failures_keep_no_transport_text(raised, error):
    adapter = OpenAIProviderAdapter(api_key="sk-test")
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        post.side_effect = raised
        with pytest.raises(error) as caught:
            await adapter.generate(REQUEST, DECISION)

    assert carries_no_marker(caught.value)


class BuggyAdapter(ModelProviderAdapter):
    """An adapter whose failure isn't a gateway error and quotes the prompt."""

    def __init__(self, name: str, gateway_error: bool = False):
        self._name = name
        self._gateway_error = gateway_error

    @property
    def name(self) -> str:
        return self._name

    def supports_model(self, model: str) -> bool:
        return True

    def _fail(self, request: ModelRequest) -> Exception:
        text = f"failed on {request.messages[0].content}"
        return ProviderUnavailableError(text) if self._gateway_error else RuntimeError(text)

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        raise self._fail(request)

    async def generate_stream(
        self, request: ModelRequest, decision: ModelDecision
    ) -> AsyncIterator[ModelStreamChunk]:
        raise self._fail(request)
        yield  # pragma: no cover - makes this an async generator


def policy_request(*providers: str) -> ModelRequest:
    return ModelRequest(
        messages=REQUEST.messages,
        context=CONTEXT,
        policy=ModelPolicy(
            preferred_provider=providers[0], fallback_providers=list(providers[1:]), max_retries=0
        ),
    )


def logged_text(caplog) -> str:
    return "\n".join(r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_an_adapter_bug_is_replaced_by_a_fixed_message(caplog):
    gateway = ModelGateway(providers=[BuggyAdapter("buggy")], approved_providers={"buggy"})

    with caplog.at_level(logging.DEBUG), pytest.raises(ModelGatewayError) as caught:
        await gateway.generate(policy_request("buggy"))

    assert carries_no_marker(caught.value)
    assert MARKER not in logged_text(caplog)


@pytest.mark.asyncio
async def test_the_gateway_logs_types_never_messages(caplog):
    # Even a gateway error whose adapter put the prompt in its message.
    gateway = ModelGateway(
        providers=[BuggyAdapter("a", gateway_error=True), BuggyAdapter("b")],
        approved_providers={"a", "b"},
    )

    with caplog.at_level(logging.DEBUG), pytest.raises(FallbackExhaustedError) as caught:
        await gateway.generate(policy_request("a", "b"))

    assert MARKER not in logged_text(caplog)
    assert carries_no_marker(caught.value)
    assert "ProviderUnavailableError" in logged_text(caplog)


@pytest.mark.asyncio
async def test_a_streaming_adapter_bug_is_replaced_by_a_fixed_message(caplog):
    gateway = ModelGateway(providers=[BuggyAdapter("buggy")], approved_providers={"buggy"})

    with caplog.at_level(logging.DEBUG), pytest.raises(ModelGatewayError) as caught:
        async for _ in gateway.generate_stream(policy_request("buggy")):
            pass

    assert carries_no_marker(caught.value)
    assert MARKER not in logged_text(caplog)
