import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.services.ai import dependencies
from app.services.ai.deterministic import DeterministicLLM
from app.services.ai.openai import OpenAILLM

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


def test_get_llm_follows_the_setting(monkeypatch):
    monkeypatch.setattr(dependencies.settings, "MODEL_PROVIDER", "deterministic")
    assert isinstance(dependencies.get_llm(), DeterministicLLM)
    monkeypatch.setattr(dependencies.settings, "MODEL_PROVIDER", "openai")
    assert isinstance(dependencies.get_llm(), OpenAILLM)


async def test_the_deterministic_answer_identifies_its_prompt():
    llm = DeterministicLLM()
    first = await llm.generate("explain photosynthesis")
    assert first == await llm.generate("explain photosynthesis")
    assert first != await llm.generate("explain osmosis")
    assert first.startswith("[deterministic model] prompt sha256:")
