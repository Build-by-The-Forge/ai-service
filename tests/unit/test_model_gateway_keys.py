"""AI-01 PR 1, F3: no placeholder key, and a strict start refuses a hollow approval."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

import app.core.models.providers.openai as openai_module
from app.core.config import Settings
from app.core.models.contracts import (
    ModelDecision,
    ModelInvocationContext,
    ModelMessage,
    ModelRequest,
    ModelRole,
)
from app.core.models.errors import AuthenticationError
from app.core.models.providers.openai import OpenAIProviderAdapter

SECRETS = {"BACKEND_TO_AI_SECRET": "b2a-" + "x" * 40, "AI_TO_BACKEND_SECRET": "a2b-" + "z" * 40}
CALLBACK_URL = "https://api.build.example/content/ingestion/callbacks"
REQUEST = ModelRequest(
    messages=[ModelMessage(role=ModelRole.USER, content="Hello")],
    context=ModelInvocationContext(
        tenant_id="school_1", execution_id="exec_1", task_id="task_1", correlation_id="corr_1"
    ),
)
DECISION = ModelDecision(
    provider="openai",
    model="gpt-4o-mini",
    fallback_providers=[],
    max_tokens=10,
    timeout_seconds=5.0,
    max_retries=1,
    max_cost_usd=None,
    temperature=0.0,
)


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
    for name in ("OPENAI_API_KEY", "MODEL_PROVIDERS_APPROVED"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.asyncio
async def test_without_a_key_the_adapter_sends_nothing(monkeypatch):
    monkeypatch.setattr(openai_module, "settings", make("development"))
    adapter = OpenAIProviderAdapter()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        with pytest.raises(AuthenticationError):
            await adapter.generate(REQUEST, DECISION)
        with pytest.raises(AuthenticationError):
            async for _ in adapter.generate_stream(REQUEST, DECISION):
                pass

    post.assert_not_called()


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_a_strict_start_refuses_an_approved_provider_without_a_key(environment):
    with pytest.raises(ValidationError, match="OPENAI_API_KEY is not set"):
        make(environment, MODEL_PROVIDERS_APPROVED="openai")


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_a_strict_start_refuses_a_provider_with_no_adapter(environment):
    with pytest.raises(ValidationError, match="no production adapter: gemini"):
        make(environment, MODEL_PROVIDERS_APPROVED="openai,gemini", OPENAI_API_KEY="sk-real")


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_a_strict_start_accepts_an_empty_list_and_a_keyed_approval(environment):
    assert make(environment).model_providers_approved == frozenset()
    keyed = make(environment, MODEL_PROVIDERS_APPROVED="openai", OPENAI_API_KEY="sk-real")
    assert keyed.model_providers_approved == {"openai"}


@pytest.mark.parametrize("environment", ["development", "testing"])
def test_outside_strict_environments_the_adapter_refuses_at_call_time(environment):
    # Development may approve a provider before a key exists; nothing is sent.
    assert make(environment, MODEL_PROVIDERS_APPROVED="openai").model_providers_approved == {
        "openai"
    }
