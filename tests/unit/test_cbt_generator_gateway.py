"""AI-10's generator runs through the gateway (AI-01 decision P3-5).

CBTAIGenerator moved off LLMProvider: its one model call is the gateway's,
with provider approval, the cbt_exam budget, the verified request's context
and sanitized errors. The CBT path keeps no direct model path.
"""

import ast
import logging
import time
from pathlib import Path

import pytest

import app.services.ai.gateway_llm as gateway_llm
from app.core.models.budget import BudgetTracker
from app.core.models.contracts import (
    ModelDecision,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelUsage,
)
from app.core.models.errors import (
    ModelGatewayError,
    NoEligibleProviderError,
    ProviderUnavailableError,
)
from app.core.models.gateway import ModelGateway
from app.core.models.provider import ModelProviderAdapter
from app.core.service_signature_v1 import VerifiedCall
from app.services.ai.gateway_llm import TUTOR_SYSTEM, GatewayLLM, invocation_context
from app.services.generation.cbt_ai_generator import CBTAIGenerator
from app.services.generation.fusion_engine import CBTFusionEngine
from app.services.rag.retrieval import Retrieved
from tests.signing import TENANT_A, user_context

MARKER = "provider-said-5e2a"
GENERATION = Path(__file__).resolve().parents[2] / "app" / "services" / "generation"


def call() -> VerifiedCall:
    return VerifiedCall(
        sender="build-backend",
        timestamp=int(time.time()),
        request_id="5b1c9a2e-7d4f-4e8a-9c3b-1a2b3c4d5e6f",
        context=user_context(tenant_id=TENANT_A, permissions=("ai.use",)),
        raw_context=b"",
    )


class Recording(ModelProviderAdapter):
    """Answers with fixed content, recording every request; can fail."""

    def __init__(self, content: str = "1. Which is a root of x^2 = 4?", failures: int = 0):
        self.content, self.failures = content, failures
        self.requests: list[ModelRequest] = []

    @property
    def name(self) -> str:
        return "rec"

    def supports_model(self, model: str) -> bool:
        return True

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        self.requests.append(request)
        if self.failures:
            self.failures -= 1
            raise ProviderUnavailableError(f"down: {MARKER}", provider="rec")
        return ModelResponse(
            content=self.content,
            model=decision.model,
            provider="rec",
            role=ModelRole.ASSISTANT,
            usage=ModelUsage(total_tokens=10),
            context=request.context,
        )

    async def generate_stream(self, request, decision):  # pragma: no cover - not used
        raise NotImplementedError
        yield


class Retriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, frozenset[str]]] = []

    async def retrieve(self, query: str, denied_assets: frozenset[str]) -> Retrieved:
        self.calls.append((query, denied_assets))
        return Retrieved(items=[{"text": f"passage on {query}"}], sources_used=())


class Adaptive:
    async def get_difficulty(self, user_id: str, topic: str) -> str:
        return "easy"


@pytest.fixture(autouse=True)
def configured_provider(monkeypatch):
    monkeypatch.setattr(gateway_llm.settings, "MODEL_PROVIDER", "rec")


def gateway(adapter: Recording, approved: set[str] | None = None) -> ModelGateway:
    return ModelGateway(
        providers=[adapter],
        approved_providers={"rec"} if approved is None else approved,
        budget_tracker=BudgetTracker(),
    )


@pytest.mark.asyncio
async def test_the_exam_comes_through_the_gateway_with_the_verified_context():
    adapter, retriever = Recording(), Retriever()
    invocation = invocation_context(call(), "cbt_exam")

    result = await CBTAIGenerator(
        llm=GatewayLLM(gateway(adapter)), retriever=retriever
    ).generate_exam_questions(
        "math",
        "algebra",
        denied_assets=frozenset({"asset_denied"}),
        invocation=invocation,
        difficulty="hard",
        count=2,
    )

    assert result == {
        "questions": "1. Which is a root of x^2 = 4?",
        "context_used": [{"text": "passage on algebra"}],
    }
    [request] = adapter.requests
    assert request.context is invocation
    assert request.context.tenant_id == TENANT_A and request.context.task_id == "cbt_exam"
    assert [m.role for m in request.messages] == [ModelRole.SYSTEM, ModelRole.USER]
    assert request.messages[0].content == TUTOR_SYSTEM
    assert "math - algebra" in request.messages[1].content
    assert "passage on algebra" in request.messages[1].content
    # The configured provider only, no fallbacks: approval is the gateway's.
    assert request.policy.preferred_provider == "rec" and request.policy.fallback_providers == []
    # Denied assets still reach retrieval.
    assert retriever.calls == [("algebra", frozenset({"asset_denied"}))]


@pytest.mark.asyncio
async def test_with_no_approved_provider_nothing_is_called():
    adapter = Recording()

    with pytest.raises(NoEligibleProviderError):
        await CBTAIGenerator(
            llm=GatewayLLM(gateway(adapter, approved=set())), retriever=Retriever()
        ).generate_exam_questions(
            "math",
            "algebra",
            denied_assets=frozenset(),
            invocation=invocation_context(call(), "cbt_exam"),
        )

    assert adapter.requests == []


@pytest.mark.asyncio
async def test_it_runs_under_the_cbt_exam_budget(monkeypatch):
    seen: list[str] = []
    original = gateway_llm.capability_limits

    def spy(capability):
        seen.append(capability)
        return original(capability)

    monkeypatch.setattr(gateway_llm, "capability_limits", spy)
    model = gateway(Recording())

    await CBTAIGenerator(llm=GatewayLLM(model), retriever=Retriever()).generate_exam_questions(
        "math",
        "algebra",
        denied_assets=frozenset(),
        invocation=invocation_context(call(), "cbt_exam"),
    )

    assert seen == ["cbt_exam"]
    # The scope ended with the request: nothing is left behind.
    assert model.budget_tracker.tracked_scope_count == 0


@pytest.mark.asyncio
async def test_without_the_requests_context_nothing_is_retrieved_or_sent():
    adapter, retriever = Recording(), Retriever()

    with pytest.raises(ValueError, match="needs its request's context"):
        await CBTAIGenerator(
            llm=GatewayLLM(gateway(adapter)), retriever=retriever
        ).generate_exam_questions(
            "math", "algebra", denied_assets=frozenset(), invocation=None  # type: ignore[arg-type]
        )

    assert retriever.calls == [] and adapter.requests == []


@pytest.mark.asyncio
async def test_a_provider_failure_reaches_the_caller_without_its_text(caplog):
    adapter = Recording(failures=5)

    with caplog.at_level(logging.DEBUG), pytest.raises(ModelGatewayError) as caught:
        await CBTAIGenerator(
            llm=GatewayLLM(gateway(adapter)), retriever=Retriever()
        ).generate_exam_questions(
            "math",
            "algebra",
            denied_assets=frozenset(),
            invocation=invocation_context(call(), "cbt_exam"),
        )

    assert MARKER not in str(caught.value) and MARKER not in repr(caught.value.args)
    assert all(MARKER not in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_the_fusion_engine_generates_through_the_gateway():
    adapter = Recording()
    invocation = invocation_context(call(), "cbt_exam")
    engine = CBTFusionEngine(
        ai_generator=CBTAIGenerator(llm=GatewayLLM(gateway(adapter)), retriever=Retriever()),
        adaptive_engine=Adaptive(),
    )

    result = await engine.generate_personalized_exam(
        "user-1", "math", "algebra", denied_assets=frozenset(), invocation=invocation
    )

    assert result["difficulty"] == "easy"
    [request] = adapter.requests
    assert request.context is invocation


# --- no direct model path ----------------------------------------------------------------

LEGACY_MODULES = {"app.services.ai.llm", "app.services.ai.openai", "app.services.ai.deterministic"}
DIRECT_PROVIDERS = {"openai", "app.services.ai.openai", "app.services.ai.deterministic"}
DIRECT_NAMES = {"OpenAILLM", "DeterministicLLM", "AsyncOpenAI"}


def imports(path: Path) -> tuple[set[str], set[str]]:
    """The modules a file imports, and every name it imports or uses."""
    modules: set[str] = set()
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return modules, names


@pytest.mark.parametrize("module", ["cbt_ai_generator.py", "fusion_engine.py"])
def test_the_cbt_path_imports_nothing_from_the_old_model_layer(module):
    modules, names = imports(GENERATION / module)

    assert not modules & LEGACY_MODULES
    assert "LLMProvider" not in names and not names & DIRECT_NAMES


def test_no_generation_module_reaches_a_provider_directly():
    for path in sorted(GENERATION.glob("*.py")):
        modules, names = imports(path)
        assert not {
            m for m in modules if m.split(".")[0] == "openai" or m in DIRECT_PROVIDERS
        }, path.name
        assert not names & DIRECT_NAMES, path.name
