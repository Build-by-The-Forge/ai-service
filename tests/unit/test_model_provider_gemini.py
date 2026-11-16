"""The Gemini adapter, through the real google-genai SDK over httpx.MockTransport.

No network: every request the SDK makes is answered here and recorded, so
the tests check what Gemini would actually be sent, what comes back, and
that the gateway keeps control of keys, retries, timeouts and error text.
"""

import json

import httpx
import pytest

import app.core.models.providers.gemini as gemini_module
from app.core.models.contracts import (
    ModelDecision,
    ModelInvocationContext,
    ModelMessage,
    ModelRequest,
    ModelRole,
    StructuredOutput,
)
from app.core.models.errors import (
    AuthenticationError,
    InvalidRequestError,
    ModelGatewayError,
    ModelTimeoutError,
    ProviderUnavailableError,
    RateLimitExceededError,
)
from app.core.models.providers.gemini import GeminiProviderAdapter, calculate_gemini_cost

SECRET = "provider-body-7c41"
CONTEXT = ModelInvocationContext(
    tenant_id="tenant_a", execution_id="exec_1", task_id="chat", correlation_id="corr_1"
)
DECISION = ModelDecision(
    provider="gemini",
    model="gemini-3.8-flash",
    fallback_providers=[],
    max_tokens=256,
    timeout_seconds=12.5,
    max_retries=1,
    max_cost_usd=None,
    temperature=0.3,
)
ANSWER = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {"text": "Osmosis moves water [1]."},
                    {"text": "hidden", "thought": True},
                ],
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 7,
        "candidatesTokenCount": 3,
        "thoughtsTokenCount": 5,
        "totalTokenCount": 15,
    },
}


def request(**changes) -> ModelRequest:
    values = {
        "messages": [
            ModelMessage(role=ModelRole.SYSTEM, content="You are an expert AI tutor."),
            ModelMessage(role=ModelRole.USER, content="Explain osmosis."),
            ModelMessage(role=ModelRole.ASSISTANT, content="Shall I use the notes?"),
            ModelMessage(role=ModelRole.USER, content="Yes."),
        ],
        "context": CONTEXT,
    }
    return ModelRequest(**(values | changes))


class Provider:
    """A Gemini endpoint stand-in: answers each request with `respond`."""

    def __init__(self, respond):
        self.respond = respond
        self.requests: list[httpx.Request] = []

    def client(self) -> httpx.AsyncClient:
        def handler(req: httpx.Request) -> httpx.Response:
            self.requests.append(req)
            return self.respond(req)

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def adapter(self, api_key: str | None = "g-test-key") -> GeminiProviderAdapter:
        return GeminiProviderAdapter(api_key=api_key, http_client=self.client())


def ok(req: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=ANSWER)


@pytest.fixture(autouse=True)
def no_configured_key(monkeypatch):
    monkeypatch.setattr(gemini_module.settings, "GEMINI_API_KEY", None)


# --- what Gemini is sent -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_request_is_gemini_shaped_with_the_explicit_key():
    provider = Provider(ok)

    await provider.adapter().generate(request(), DECISION)

    [sent] = provider.requests
    assert str(sent.url) == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"
    )
    assert sent.headers["x-goog-api-key"] == "g-test-key"
    body = json.loads(sent.content)
    assert body["systemInstruction"]["parts"] == [{"text": "You are an expert AI tutor."}]
    assert [(c["role"], c["parts"][0]["text"]) for c in body["contents"]] == [
        ("user", "Explain osmosis."),
        ("model", "Shall I use the notes?"),
        ("user", "Yes."),
    ]
    assert body["generationConfig"] == {"temperature": 0.3, "maxOutputTokens": 256}
    # The attempt's timeout, which the gateway caps at the deadline's time left.
    assert sent.extensions["timeout"]["read"] == pytest.approx(12.5)


@pytest.mark.asyncio
async def test_a_structured_request_asks_for_json():
    provider = Provider(ok)

    await provider.adapter().generate(
        request(structured=StructuredOutput(kind="questions", hints={"count": 2})), DECISION
    )

    body = json.loads(provider.requests[0].content)
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert "count" not in json.dumps(body)  # the hints are the offline adapter's, not sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"tools": [{"type": "function", "function": {"name": "lookup"}}]},
        {"messages": [ModelMessage(role=ModelRole.TOOL, content="result")]},
    ],
    ids=["tools", "tool message"],
)
async def test_what_the_adapter_cant_carry_is_refused_before_sending(changes):
    provider = Provider(ok)

    with pytest.raises(InvalidRequestError):
        await provider.adapter().generate(request(**changes), DECISION)

    assert provider.requests == []


# --- keys --------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_without_a_key_nothing_is_sent_whatever_the_environment_says(monkeypatch):
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(name, "env-key-never-used")
    provider = Provider(ok)

    with pytest.raises(AuthenticationError):
        await provider.adapter(api_key=None).generate(request(), DECISION)
    with pytest.raises(AuthenticationError):
        async for _ in provider.adapter(api_key=None).generate_stream(request(), DECISION):
            pass

    assert provider.requests == []


@pytest.mark.asyncio
async def test_the_sdks_environment_never_redirects_the_call(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key-never-used")
    provider = Provider(ok)

    await provider.adapter().generate(request(), DECISION)

    [sent] = provider.requests
    assert sent.url.host == "generativelanguage.googleapis.com"
    assert sent.headers["x-goog-api-key"] == "g-test-key"


# --- the answer --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_answer_its_usage_and_cost():
    response = await Provider(ok).adapter().generate(request(), DECISION)

    assert response.content == "Osmosis moves water [1]."  # no thought parts
    assert (response.provider, response.model, response.finish_reason) == (
        "gemini",
        "gemini-3.8-flash",
        "stop",
    )
    # Thinking tokens are billed as output.
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (7, 8)
    assert response.usage.total_tokens == 15
    assert response.usage.estimated_cost_usd == calculate_gemini_cost("gemini-3.8-flash", 7, 8)
    assert response.context is CONTEXT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        {"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []},
        {"candidates": [{"content": {"parts": [{"text": SECRET}]}, "finishReason": "SAFETY"}]},
        {"candidates": []},
        {"candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "STOP"}]},
        {"candidates": [{"finishReason": "STOP"}]},
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": SECRET, "thought": True}]},
                    "finishReason": "MAX_TOKENS",
                }
            ]
        },
        {"candidates": [{"content": {"parts": [{"text": " \n "}]}, "finishReason": "STOP"}]},
    ],
    ids=[
        "blocked prompt",
        "withheld answer",
        "no answer",
        "empty candidate",
        "candidate without content",
        "thought-only candidate",
        "whitespace-only candidate",
    ],
)
async def test_a_blocked_or_empty_answer_fails_without_retry_or_text(answer):
    provider = Provider(lambda req: httpx.Response(200, json=answer))

    with pytest.raises(ModelGatewayError) as caught:
        await provider.adapter().generate(request(), DECISION)

    assert caught.value.retryable is False
    assert SECRET not in str(caught.value)


def test_the_cost_estimate_is_high_not_low():
    # The paid Standard tier's 2027 prices: $1.50 in, $7.50 out, per 1M tokens.
    assert calculate_gemini_cost("gemini-3.8-flash", 1_000_000, 1_000_000) == 9.0
    # An unlisted model is estimated at the highest listed price.
    assert calculate_gemini_cost("gemini-unknown", 1000, 1000) == calculate_gemini_cost(
        "gemini-3.8-flash", 1000, 1000
    )


# --- failures ----------------------------------------------------------------------------


def status(code: int):
    def respond(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            code, json={"error": {"code": code, "message": f"refused: {SECRET}", "status": "X"}}
        )

    return respond


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,error,retryable",
    [
        (400, InvalidRequestError, False),
        (404, InvalidRequestError, False),
        (401, AuthenticationError, False),
        (403, AuthenticationError, False),
        (429, RateLimitExceededError, True),
        (500, ProviderUnavailableError, True),
        (503, ProviderUnavailableError, True),
        (418, ModelGatewayError, False),
    ],
)
async def test_a_provider_error_is_mapped_by_status_with_none_of_its_text(code, error, retryable):
    provider = Provider(status(code))

    with pytest.raises(error) as caught:
        await provider.adapter().generate(request(), DECISION)

    failure = caught.value
    assert type(failure) is error and failure.status_code == code
    assert failure.retryable is retryable
    # Nothing of the provider's body: not in the message, the arguments, or
    # attached as the original exception.
    assert SECRET not in str(failure) and SECRET not in repr(failure.args)
    assert failure.__context__ is None and failure.__cause__ is None
    # One attempt: the SDK doesn't retry; the gateway decides.
    assert len(provider.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raised,error",
    [
        (httpx.ReadTimeout(f"slow {SECRET}"), ModelTimeoutError),
        (httpx.ConnectError(f"no route {SECRET}"), ProviderUnavailableError),
    ],
    ids=["timeout", "network"],
)
async def test_a_timeout_or_network_failure_is_mapped_by_type(raised, error):
    def respond(req: httpx.Request) -> httpx.Response:
        raise raised

    with pytest.raises(error) as caught:
        await Provider(respond).adapter().generate(request(), DECISION)

    assert SECRET not in str(caught.value) and caught.value.__context__ is None


# --- streaming ---------------------------------------------------------------------------


def sse(*chunks: dict) -> httpx.Response:
    body = "".join(f"data: {json.dumps(chunk)}\r\n\r\n" for chunk in chunks)
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


@pytest.mark.asyncio
async def test_a_stream_yields_text_and_ends_with_its_finish_reason():
    provider = Provider(
        lambda req: sse(
            {"candidates": [{"content": {"parts": [{"text": "Osmosis "}]}}]},
            {
                "candidates": [
                    {"content": {"parts": [{"text": "moves water."}]}, "finishReason": "STOP"}
                ]
            },
        )
    )

    chunks = [c async for c in provider.adapter().generate_stream(request(), DECISION)]

    assert [c.delta_content for c in chunks] == ["Osmosis ", "moves water."]
    assert [c.finish_reason for c in chunks] == [None, "stop"]
    assert ":streamGenerateContent" in str(provider.requests[0].url)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks",
    [
        ({"promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": SECRET}},),
        (
            {"usageMetadata": {"promptTokenCount": 7}},
            {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}, "candidates": []},
        ),
    ],
    ids=["blocked prompt", "blocked after a usage-only chunk"],
)
async def test_a_stream_with_a_blocked_prompt_fails_without_retry_or_text(chunks):
    provider = Provider(lambda req: sse(*chunks))
    delivered = []

    with pytest.raises(ModelGatewayError) as caught:
        async for chunk in provider.adapter().generate_stream(request(), DECISION):
            delivered.append(chunk)

    assert delivered == []
    # The block itself, not only a stream that came back empty.
    assert str(caught.value) == "Gemini blocked the prompt"
    assert caught.value.retryable is False
    assert SECRET not in str(caught.value) and caught.value.__context__ is None


def thought(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text, "thought": True}]}}]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks",
    [
        (
            thought(SECRET),
            {"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]},
        ),
        (
            thought(SECRET),
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": SECRET, "thought": True}]},
                        "finishReason": "MAX_TOKENS",
                    }
                ]
            },
        ),
        ({"candidates": [{"content": {"parts": [{"text": "  "}]}, "finishReason": "STOP"}]},),
        (thought(SECRET), {"candidates": [{"content": {"parts": []}}]}),
        ({"usageMetadata": {"promptTokenCount": 7}},),
    ],
    ids=[
        "thinking then an empty finish",
        "thought-only finish",
        "whitespace-only finish",
        "ends without text or finish",
        "usage only",
    ],
)
async def test_a_stream_without_visible_text_fails_and_never_finishes(chunks):
    provider = Provider(lambda req: sse(*chunks))
    delivered = []

    with pytest.raises(ModelGatewayError) as caught:
        async for chunk in provider.adapter().generate_stream(request(), DECISION):
            delivered.append(chunk)

    assert str(caught.value) == "Gemini returned no answer"
    assert caught.value.retryable is False and caught.value.__context__ is None
    # No finish was reported, and nothing of the thoughts was delivered.
    assert all(c.finish_reason is None for c in delivered)
    assert all(SECRET not in c.delta_content for c in delivered)


@pytest.mark.asyncio
async def test_a_stream_skips_thinking_and_delivers_the_answer():
    provider = Provider(
        lambda req: sse(
            thought(SECRET),
            {"candidates": [{"content": {"parts": [{"text": "Osmosis "}]}}]},
            thought(SECRET),
            {
                "candidates": [
                    {"content": {"parts": [{"text": "moves water."}]}, "finishReason": "STOP"}
                ]
            },
        )
    )

    chunks = [c async for c in provider.adapter().generate_stream(request(), DECISION)]

    assert [(c.delta_content, c.finish_reason) for c in chunks] == [
        ("Osmosis ", None),
        ("moves water.", "stop"),
    ]


@pytest.mark.asyncio
async def test_a_stream_failure_is_mapped_with_none_of_its_text():
    with pytest.raises(ProviderUnavailableError) as caught:
        async for _ in Provider(status(503)).adapter().generate_stream(request(), DECISION):
            pass

    assert SECRET not in str(caught.value) and caught.value.__context__ is None
