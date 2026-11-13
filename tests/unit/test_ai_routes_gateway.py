"""AI-01 PR 3: /v1/ai/chat, /explain and /generate-questions through the gateway.

In deterministic mode the routes answer exactly as they did through
DeterministicLLM. A model execution failure is a 503 with a fixed body
(decision P3-1); with no approved provider they fail closed (P3-2); invalid
input is still a 422 that reaches no model; and the request's model context
comes from the verified signed call (P3-4).
"""

import json
import os

import pytest
from httpx import ASGITransport, AsyncClient

import app.services.ai.gateway_llm as gateway_llm
from app.core.models.budget import BudgetTracker
from app.core.models.contracts import ModelDecision, ModelRequest, ModelResponse, ModelRole
from app.core.models.errors import ProviderUnavailableError
from app.core.models.gateway import ModelGateway
from app.core.models.provider import ModelProviderAdapter
from app.core.models.providers.deterministic import DeterministicProviderAdapter
from app.core.replay_guard import RedisReplayGuard
from app.core.service_signature_v1 import REQUEST_ID_HEADER
from app.main import create_app
from app.services.ai.deterministic import DeterministicLLM
from app.services.ai.dependencies import get_ai_orchestrator
from app.services.ai.gateway_llm import GatewayLLM
from app.services.ai.llm import LLMProvider
from app.services.ai.orchestrator import AIOrchestrator
from app.services.ai.tenanted_retriever import TenantAwareRetrievalService
from app.services.rag.scopes import get_scope_store
from tests.rag_support import institutional
from tests.signing import TENANT_A, signed_headers, user_context

REDIS_URL = os.getenv("TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL not set")

NONCE = "Zt3m0VbqO9aX1yR4kLw2pQ8sHn6uJc5d"
MARKER = "student-passage-a27e"
UNAVAILABLE = {
    "success": False,
    "code": "AI_SERVICE_UNAVAILABLE",
    "message": "AI service unavailable",
    "data": {},
}
# path, permission, a valid body
ROUTES = [
    ("/v1/ai/chat", "ai.use", {"message": "What is osmosis?"}),
    ("/v1/ai/explain", "ai.use", {"subject": "osmosis"}),
    ("/v1/ai/generate-questions", "ai.questions.generate", {"topic": "osmosis", "count": 2}),
]
ROUTE_IDS = [r[0] for r in ROUTES]


class NoMemory:
    async def get_user_memory(self, user_id: str):
        return []


class Recording(ModelProviderAdapter):
    """Answers with fixed content, recording each request; can always fail."""

    def __init__(self, content: str = "Water moves [1].", fail: bool = False):
        self.content, self.fail = content, fail
        self.requests: list[ModelRequest] = []

    @property
    def name(self) -> str:
        return "rec"

    def supports_model(self, model: str) -> bool:
        return True

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        self.requests.append(request)
        if self.fail:
            raise ProviderUnavailableError(f"provider said: {MARKER}", provider="rec")
        return ModelResponse(
            content=self.content, model=decision.model, provider="rec", role=ModelRole.ASSISTANT
        )

    async def generate_stream(self, request, decision):  # pragma: no cover - not used
        raise NotImplementedError
        yield


@pytest.fixture
async def guard():
    guard = RedisReplayGuard.connect(REDIS_URL, timeout=2.0)
    yield guard
    await guard.close()


@pytest.fixture
async def notes(in_memory_retrieval):
    """A passage the caller's tenant may read, so the model is asked."""
    await in_memory_retrieval.index_document(
        tenant_id=TENANT_A,
        document_id="notes",
        document_title="Biology notes",
        text="Osmosis moves water across a semipermeable membrane toward higher solute concentration.",
    )
    await institutional(get_scope_store(), TENANT_A, "notes")


def serving(guard, llm: LLMProvider):
    app = create_app(replay_guard=guard)
    app.dependency_overrides[get_ai_orchestrator] = lambda: AIOrchestrator(
        llm=llm, retriever=TenantAwareRetrievalService(), memory=NoMemory()
    )
    return app


def through(*adapters: ModelProviderAdapter, approved: set[str] | None = None) -> GatewayLLM:
    return GatewayLLM(
        ModelGateway(
            providers=list(adapters),
            approved_providers={a.name for a in adapters} if approved is None else approved,
            budget_tracker=BudgetTracker(),
        )
    )


async def post(app, path: str, permission: str, body: dict):
    raw = json.dumps({"response_nonce": NONCE, "denied_assets": [], **body}).encode()
    headers = signed_headers(
        raw, target=path, context=user_context(tenant_id=TENANT_A, permissions=(permission,))
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://ai") as http:
        return await http.post(path, content=raw, headers=headers), headers


@pytest.mark.parametrize("path,permission,body", ROUTES, ids=ROUTE_IDS)
async def test_deterministic_mode_answers_through_the_gateway_as_before(
    guard, notes, monkeypatch, path, permission, body
):
    monkeypatch.setattr(gateway_llm.settings, "MODEL_PROVIDER", "deterministic")

    before, _ = await post(serving(guard, DeterministicLLM()), path, permission, body)
    after, _ = await post(
        serving(guard, through(DeterministicProviderAdapter())), path, permission, body
    )

    assert before.status_code == after.status_code == 200, after.text
    assert after.json()["data"] == before.json()["data"]
    assert after.json()["sources_used"] == before.json()["sources_used"]


@pytest.mark.parametrize("path,permission,body", ROUTES, ids=ROUTE_IDS)
async def test_a_model_execution_failure_is_a_503_with_a_fixed_body(
    guard, notes, monkeypatch, path, permission, body
):
    monkeypatch.setattr(gateway_llm.settings, "MODEL_PROVIDER", "rec")
    monkeypatch.setattr(
        "app.core.models.resilience.ModelRetryPolicy.compute_backoff", lambda attempt: 0.0
    )

    response, _ = await post(serving(guard, through(Recording(fail=True))), path, permission, body)

    assert response.status_code == 503
    assert response.json() == UNAVAILABLE
    assert MARKER not in response.text


@pytest.mark.parametrize("path,permission,body", ROUTES, ids=ROUTE_IDS)
async def test_with_no_approved_provider_the_routes_fail_closed(
    guard, notes, monkeypatch, path, permission, body
):
    monkeypatch.setattr(gateway_llm.settings, "MODEL_PROVIDER", "rec")
    unapproved = Recording()

    response, _ = await post(
        serving(guard, through(unapproved, approved=set())), path, permission, body
    )

    assert response.status_code == 503 and response.json() == UNAVAILABLE
    assert unapproved.requests == []


@pytest.mark.parametrize("path,permission,body", ROUTES, ids=ROUTE_IDS)
async def test_invalid_input_is_still_422_and_reaches_no_model(
    guard, notes, monkeypatch, path, permission, body
):
    monkeypatch.setattr(gateway_llm.settings, "MODEL_PROVIDER", "rec")
    recording = Recording()
    invalid = {k: 12345 for k in body}  # every field the wrong type

    response, _ = await post(serving(guard, through(recording)), path, permission, invalid)

    assert response.status_code == 422
    assert recording.requests == []


async def test_the_model_context_comes_from_the_verified_call(guard, notes, monkeypatch):
    monkeypatch.setattr(gateway_llm.settings, "MODEL_PROVIDER", "rec")
    recording = Recording()

    response, headers = await post(
        serving(guard, through(recording)), "/v1/ai/chat", "ai.use", {"message": "What is osmosis?"}
    )

    assert response.status_code == 200, response.text
    [request] = recording.requests
    assert request.context.tenant_id == TENANT_A
    assert request.context.correlation_id == headers[REQUEST_ID_HEADER]
    assert request.context.task_id == "chat"


async def test_an_unusable_question_answer_is_still_a_200_with_the_questions_discarded(
    guard, notes, monkeypatch
):
    monkeypatch.setattr(gateway_llm.settings, "MODEL_PROVIDER", "rec")

    response, _ = await post(
        serving(guard, through(Recording(content="not JSON"))),
        "/v1/ai/generate-questions",
        "ai.questions.generate",
        {"topic": "osmosis", "count": 2},
    )

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["questions"] == [] and data["discarded"] == 2
