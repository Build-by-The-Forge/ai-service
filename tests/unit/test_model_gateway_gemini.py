"""Gemini behind the gateway's approval boundary (G-33).

Registered is not eligible: Gemini is called only when MODEL_PROVIDERS_APPROVED
names it. When it is, chat, explain and question generation reach it through
GatewayLLM with its own model, and the gateway, not the SDK, retries.
"""

import json
import time

import httpx
import pytest

import app.core.models.gateway as gateway_module
import app.services.ai.gateway_llm as gateway_llm
from app.core.models.budget import BudgetTracker
from app.core.models.errors import NoEligibleProviderError
from app.core.models.gateway import ModelGateway
from app.core.models.providers.gemini import GeminiProviderAdapter
from app.core.models.providers.openai import OpenAIProviderAdapter
from app.core.service_signature_v1 import VerifiedCall
from app.services.ai.gateway_llm import GatewayLLM, invocation_context
from tests.signing import TENANT_A, user_context

ANSWER = {
    "candidates": [
        {"content": {"parts": [{"text": "Osmosis moves water."}]}, "finishReason": "STOP"}
    ],
    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 4, "totalTokenCount": 9},
}


def call() -> VerifiedCall:
    return VerifiedCall(
        sender="build-backend",
        timestamp=int(time.time()),
        request_id="2f0e8c1a-6b4d-4e3a-9c7b-5d1e2f3a4b5c",
        context=user_context(tenant_id=TENANT_A, permissions=("ai.use",)),
        raw_context=b"",
    )


class Endpoint:
    def __init__(self, *statuses: int):
        self.statuses = list(statuses)
        self.requests: list[httpx.Request] = []

    def adapter(self) -> GeminiProviderAdapter:
        def handler(req: httpx.Request) -> httpx.Response:
            self.requests.append(req)
            code = self.statuses.pop(0) if self.statuses else 200
            if code != 200:
                return httpx.Response(code, json={"error": {"code": code, "message": "x"}})
            return httpx.Response(200, json=ANSWER)

        return GeminiProviderAdapter(
            api_key="g-test-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )


@pytest.fixture(autouse=True)
def gemini_configured(monkeypatch):
    monkeypatch.setattr(gateway_llm.settings, "MODEL_PROVIDER", "gemini")
    monkeypatch.setattr(gateway_module.settings, "GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setattr(
        "app.core.models.resilience.ModelRetryPolicy.compute_backoff", lambda attempt: 0.0
    )


def through(endpoint: Endpoint, approved: set[str]) -> GatewayLLM:
    return GatewayLLM(
        ModelGateway(
            providers=[endpoint.adapter()],
            approved_providers=approved,
            budget_tracker=BudgetTracker(),
        )
    )


@pytest.mark.asyncio
async def test_unapproved_gemini_is_never_called():
    endpoint = Endpoint()

    with through(endpoint, approved=set()).for_capability(
        "chat", invocation_context(call(), "chat")
    ) as llm:
        with pytest.raises(NoEligibleProviderError):
            await llm.generate("Explain osmosis.")

    assert endpoint.requests == []


@pytest.mark.asyncio
async def test_approved_gemini_answers_through_the_gateway_with_its_own_model():
    endpoint = Endpoint()

    with through(endpoint, approved={"gemini"}).for_capability(
        "chat", invocation_context(call(), "chat")
    ) as llm:
        answer = await llm.generate("Explain osmosis.")

    assert answer == "Osmosis moves water."
    [sent] = endpoint.requests
    assert sent.url.path.endswith("/models/gemini-3.8-flash:generateContent")
    body = json.loads(sent.content)
    assert body["contents"][-1]["parts"][0]["text"] == "Explain osmosis."


@pytest.mark.asyncio
async def test_the_gateway_retries_a_503_not_the_sdk():
    endpoint = Endpoint(503)

    with through(endpoint, approved={"gemini"}).for_capability(
        "chat", invocation_context(call(), "chat")
    ) as llm:
        assert await llm.generate("Explain osmosis.") == "Osmosis moves water."

    # The SDK's one attempt, then the gateway's one retry (max_retries=1).
    assert len(endpoint.requests) == 2


@pytest.mark.asyncio
async def test_question_generation_asks_gemini_for_json():
    endpoint = Endpoint()
    endpoint_answer = {"questions": []}

    def handler(req: httpx.Request) -> httpx.Response:
        endpoint.requests.append(req)
        content = {"candidates": [{"content": {"parts": [{"text": json.dumps(endpoint_answer)}]}}]}
        return httpx.Response(200, json=content)

    llm = GatewayLLM(
        ModelGateway(
            providers=[
                GeminiProviderAdapter(
                    api_key="g-test-key",
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                )
            ],
            approved_providers={"gemini"},
            budget_tracker=BudgetTracker(),
        )
    )
    context = invocation_context(call(), "generate_questions")

    with llm.for_capability("generate_questions", context) as bound:
        assert (
            await bound.generate_json("Write questions.", kind="questions", count=2)
            == endpoint_answer
        )

    body = json.loads(endpoint.requests[0].content)
    assert body["generationConfig"]["responseMimeType"] == "application/json"


def test_the_default_gateway_registers_gemini_but_approves_nothing(monkeypatch):
    monkeypatch.setattr(gateway_module.settings, "MODEL_PROVIDER", "openai")
    monkeypatch.setattr(gateway_module.settings, "MODEL_PROVIDERS_APPROVED", "")

    gateway = ModelGateway()

    assert isinstance(gateway.get_provider("gemini"), GeminiProviderAdapter)
    assert isinstance(gateway.get_provider("openai"), OpenAIProviderAdapter)
    assert gateway._approved == frozenset()
