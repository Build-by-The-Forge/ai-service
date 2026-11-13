import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.services.ai import dependencies
from app.services.ai.deterministic import DeterministicLLM
from app.services.ai.gateway_llm import GatewayLLM

CALLBACK_URL = "https://api.build.example/content/ingestion/callbacks"
SECRETS = {"BACKEND_TO_AI_SECRET": "b2a-" + "x" * 40, "AI_TO_BACKEND_SECRET": "a2b-" + "z" * 40}


def make(environment: str, **values) -> Settings:
    return Settings(
        _env_file=None,
        APP_ENVIRONMENT=environment,
        BACKEND_CALLBACK_URL=CALLBACK_URL,
        **SECRETS,
        **values,
    )


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)


def test_openai_is_the_default_provider():
    assert make("production").MODEL_PROVIDER == "openai"


@pytest.mark.parametrize("environment", ["development", "testing"])
def test_the_deterministic_provider_is_allowed_for_development_and_tests(environment):
    assert make(environment, MODEL_PROVIDER="deterministic").MODEL_PROVIDER == "deterministic"


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_the_deterministic_provider_is_refused_in_strict_environments(environment):
    with pytest.raises(ValidationError, match="MODEL_PROVIDER=deterministic is refused"):
        make(environment, MODEL_PROVIDER="deterministic")


async def test_chat_explain_and_questions_are_wired_to_the_gateway(monkeypatch):
    """AI-01 PR 3: the orchestrator's model layer is the gateway, whatever
    MODEL_PROVIDER says; the provider is the gateway's policy and approval."""
    gateway = object()
    monkeypatch.setattr(dependencies, "get_model_gateway", lambda: gateway)

    orchestrator = await dependencies.get_ai_orchestrator(db=None)

    assert isinstance(orchestrator.llm, GatewayLLM)
    assert orchestrator.llm._gateway is gateway
    assert not hasattr(dependencies, "get_llm")  # no path around the gateway


async def test_the_deterministic_answer_identifies_its_prompt():
    llm = DeterministicLLM()
    first = await llm.generate("explain photosynthesis")
    assert first == await llm.generate("explain photosynthesis")
    assert first != await llm.generate("explain osmosis")
    assert first.startswith("[deterministic model] prompt sha256:")


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_a_strict_environment_cannot_approve_the_deterministic_adapter(environment):
    with pytest.raises(ValidationError, match="no production adapter"):
        make(environment, MODEL_PROVIDERS_APPROVED="deterministic")


BUDGETS = [
    f"AI_BUDGET_{capability}_{limit}"
    for capability in ("CHAT", "EXPLAIN", "QUESTIONS")
    for limit in ("CALLS", "TOKENS", "COST_USD")
]


@pytest.mark.parametrize("name", BUDGETS)
@pytest.mark.parametrize("value", [0, -1, None])
def test_no_capability_runs_without_a_budget(name, value):
    """AI-01 decision P3-3: every limit is set and positive; none is unlimited."""
    with pytest.raises(ValidationError):
        make("development", **{name: value})


def test_the_budget_defaults_are_two_calls_and_conservative_caps():
    settings = make("production")
    assert (
        settings.AI_BUDGET_CHAT_CALLS,
        settings.AI_BUDGET_EXPLAIN_CALLS,
        settings.AI_BUDGET_QUESTIONS_CALLS,
    ) == (2, 2, 2)
    assert settings.AI_BUDGET_CHAT_TOKENS == settings.AI_BUDGET_EXPLAIN_TOKENS == 12_000
    assert settings.AI_BUDGET_QUESTIONS_TOKENS == 16_000
