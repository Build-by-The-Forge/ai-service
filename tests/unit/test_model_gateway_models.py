"""Each provider is asked for its own model.

The quality tiers name OpenAI models. Gemini has one configured model
(GEMINI_MODEL). A request's explicit model wins. A fallback gets its own
provider's model and is estimated at its own price, never the preferred
provider's.
"""

import pytest

import app.core.models.gateway as gateway_module
from app.core.models.budget import BudgetTracker
from app.core.models.contracts import (
    ModelDecision,
    ModelInvocationContext,
    ModelMessage,
    ModelPolicy,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelUsage,
)
from app.core.models.errors import ProviderUnavailableError
from app.core.models.gateway import ModelGateway, model_for
from app.core.models.provider import ModelProviderAdapter

CONTEXT = ModelInvocationContext(
    tenant_id="tenant_a", execution_id="exec_1", task_id="chat", correlation_id="corr_1"
)


def request(policy: ModelPolicy, model: str | None = None) -> ModelRequest:
    return ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="q")],
        context=CONTEXT,
        policy=policy,
        model=model,
    )


@pytest.fixture(autouse=True)
def gemini_model(monkeypatch):
    monkeypatch.setattr(gateway_module.settings, "GEMINI_MODEL", "gemini-test-flash")


@pytest.mark.parametrize(
    "quality,expected", [("high", "gpt-4o"), ("balanced", "gpt-4o-mini"), ("fast", "gpt-4o-mini")]
)
def test_openai_keeps_its_quality_tiers(quality, expected):
    policy = ModelPolicy(quality=quality, preferred_provider="openai")
    assert (
        ModelGateway(providers=[], approved_providers=set()).resolve_policy(request(policy)).model
        == expected
    )


@pytest.mark.parametrize("quality", ["high", "balanced", "fast"])
def test_gemini_is_asked_for_its_configured_model(quality):
    policy = ModelPolicy(quality=quality, preferred_provider="gemini")
    decision = ModelGateway(providers=[], approved_providers=set()).resolve_policy(request(policy))
    assert (decision.provider, decision.model) == ("gemini", "gemini-test-flash")


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_an_explicit_model_wins(provider):
    policy = ModelPolicy(preferred_provider=provider)
    assert model_for(provider, request(policy, model="chosen-model"), policy) == "chosen-model"


class Named(ModelProviderAdapter):
    """Records the model and provider each call was decided for; can fail."""

    def __init__(self, name: str, fail: bool = False):
        self._name, self.fail = name, fail
        self.decisions: list[ModelDecision] = []
        self.estimated: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def supports_model(self, model: str) -> bool:
        return True

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        self.estimated.append(model)
        return 0.0

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        self.decisions.append(decision)
        if self.fail:
            raise ProviderUnavailableError("down", provider=self._name)
        return ModelResponse(
            content="ok",
            model=decision.model,
            provider=self._name,
            role=ModelRole.ASSISTANT,
            usage=ModelUsage(total_tokens=1),
            context=request.context,
        )

    async def generate_stream(self, request, decision):  # pragma: no cover - not used
        raise NotImplementedError
        yield


@pytest.mark.asyncio
async def test_a_fallback_is_asked_for_its_own_model_at_its_own_price():
    gemini, openai = Named("gemini", fail=True), Named("openai")
    gateway = ModelGateway(
        providers=[gemini, openai],
        approved_providers={"gemini", "openai"},
        budget_tracker=BudgetTracker(),
    )
    policy = ModelPolicy(preferred_provider="gemini", fallback_providers=["openai"], max_retries=0)

    response = await gateway.generate(request(policy))

    assert response.provider == "openai"
    [tried] = gemini.decisions
    [answered] = openai.decisions
    assert (tried.provider, tried.model) == ("gemini", "gemini-test-flash")
    assert (answered.provider, answered.model) == ("openai", "gpt-4o-mini")
    assert gemini.estimated == ["gemini-test-flash"] and openai.estimated == ["gpt-4o-mini"]
