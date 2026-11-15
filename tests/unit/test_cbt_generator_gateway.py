"""AI-10's generator runs through the gateway (AI-01 decision P3-5).

Its one model call is the gateway's, with provider approval, the cbt_exam
budget, the verified request's context and sanitized errors. The CBT path
keeps no direct model path. (Its question correctness and grounding are
tested in test_cbt_exam_grounded.py.)
"""

import ast
import logging
from pathlib import Path

import pytest

import app.services.ai.gateway_llm as gateway_llm
from app.core import tenant
from app.core.models.contracts import ModelRole
from app.core.models.errors import ModelGatewayError, NoEligibleProviderError
from app.services.ai.gateway_llm import JSON_SYSTEM, GatewayLLM, invocation_context
from app.services.generation.cbt_ai_generator import CBTAIGenerator
from app.services.generation.fusion_engine import CBTFusionEngine
from tests.cbt_support import MARKER, Adaptive, Recording, Retriever, call, gateway
from tests.signing import TENANT_A

GENERATION = Path(__file__).resolve().parents[2] / "app" / "services" / "generation"


@pytest.fixture(autouse=True)
def configured_provider_and_scope(monkeypatch):
    monkeypatch.setattr(gateway_llm.settings, "MODEL_PROVIDER", "rec")
    tenant.set_request_scope(TENANT_A, "user-1")
    yield
    tenant.clear_request_scope()


async def generate(adapter, retriever=None, invocation="default", **kwargs):
    if invocation == "default":
        invocation = invocation_context(call(), "cbt_exam")
    return await CBTAIGenerator(
        llm=GatewayLLM(gateway(adapter, **kwargs)), retriever=retriever or Retriever()
    ).generate_exam_questions(
        "biology", "cells", denied_assets=frozenset({"asset_denied"}), invocation=invocation
    )


@pytest.mark.asyncio
async def test_the_exam_comes_through_the_gateway_with_the_verified_context():
    adapter, retriever = Recording(), Retriever()
    invocation = invocation_context(call(), "cbt_exam")

    await generate(adapter, retriever, invocation)

    [request] = adapter.requests
    assert request.context is invocation
    assert request.context.tenant_id == TENANT_A and request.context.task_id == "cbt_exam"
    # A structured request: one JSON object, checked by the gateway.
    assert [m.role for m in request.messages] == [ModelRole.SYSTEM, ModelRole.USER]
    assert request.messages[0].content == JSON_SYSTEM
    assert request.structured is not None and request.structured.kind == "questions"
    # The configured provider only, no fallbacks: approval is the gateway's.
    assert request.policy.preferred_provider == "rec" and request.policy.fallback_providers == []
    # The course disambiguates the topic; denied assets still reach retrieval.
    assert retriever.calls == [("biology cells", frozenset({"asset_denied"}))]


@pytest.mark.asyncio
async def test_with_no_approved_provider_nothing_is_called():
    adapter = Recording()

    with pytest.raises(NoEligibleProviderError):
        await generate(adapter, approved=set())

    assert adapter.requests == []


@pytest.mark.asyncio
async def test_it_runs_under_the_cbt_exam_budget(monkeypatch):
    seen: list[str] = []
    original = gateway_llm.capability_limits

    def spy(capability):
        seen.append(capability)
        return original(capability)

    monkeypatch.setattr(gateway_llm, "capability_limits", spy)

    await generate(Recording())

    assert seen == ["cbt_exam"]


@pytest.mark.asyncio
async def test_without_the_requests_context_nothing_is_retrieved_or_sent():
    adapter, retriever = Recording(), Retriever()

    with pytest.raises(ValueError, match="needs its request's context"):
        await generate(adapter, retriever, invocation=None)

    assert retriever.calls == [] and adapter.requests == []


@pytest.mark.asyncio
async def test_a_provider_failure_reaches_the_caller_without_its_text(caplog):
    with caplog.at_level(logging.DEBUG), pytest.raises(ModelGatewayError) as caught:
        await generate(Recording(failures=5))

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
        "user-1", "biology", "cells", denied_assets=frozenset(), invocation=invocation
    )

    assert result["difficulty"] == "medium"
    assert [q["prompt"] for q in result["exam"]["questions"]] == [
        "Which organelle makes ATP in cells?"
    ]
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


@pytest.mark.parametrize("module", ["cbt_ai_generator.py", "cbt_exam.py", "fusion_engine.py"])
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
