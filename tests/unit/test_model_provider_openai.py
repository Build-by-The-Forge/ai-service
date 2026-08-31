from unittest.mock import AsyncMock, patch
import httpx
import pytest

from app.core.models.contracts import (
    ModelDecision,
    ModelInvocationContext,
    ModelMessage,
    ModelRequest,
    ModelRole,
)
from app.core.models.errors import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitExceededError,
)
from app.core.models.providers.openai import (
    OpenAIProviderAdapter,
    calculate_openai_cost,
)


def test_openai_pricing_calculation():
    # gpt-4o: $0.005 / 1k input, $0.015 / 1k output
    cost = calculate_openai_cost("gpt-4o", prompt_tokens=1000, completion_tokens=1000)
    assert cost == 0.02


@pytest.mark.asyncio
async def test_openai_adapter_successful_response():
    adapter = OpenAIProviderAdapter(api_key="sk-test", base_url="https://api.openai.com/v1")
    context = ModelInvocationContext(
        tenant_id="school_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
    )
    request = ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="Hello")],
        context=context,
    )
    decision = ModelDecision(
        provider="openai",
        model="gpt-4o",
        fallback_providers=[],
        max_tokens=100,
        timeout_seconds=10.0,
        max_retries=1,
        max_cost_usd=None,
        temperature=0.7,
    )

    mock_openai_response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15,
        },
    }

    mock_http_response = httpx.Response(status_code=200, json=mock_openai_response)

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_http_response

        response = await adapter.generate(request, decision)
        assert response.content == "Hello there!"
        assert response.model == "gpt-4o"
        assert response.provider == "openai"
        assert response.usage.total_tokens == 15
        assert response.usage.estimated_cost_usd > 0


@pytest.mark.asyncio
async def test_openai_adapter_error_mappings():
    adapter = OpenAIProviderAdapter(api_key="sk-test")
    context = ModelInvocationContext(
        tenant_id="school_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
    )
    request = ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="Hello")],
        context=context,
    )
    decision = ModelDecision(
        provider="openai",
        model="gpt-4o",
        fallback_providers=[],
        max_tokens=100,
        timeout_seconds=10.0,
        max_retries=1,
        max_cost_usd=None,
        temperature=0.7,
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # Test 401 Authentication error
        mock_post.return_value = httpx.Response(status_code=401, text="Incorrect API key")
        with pytest.raises(AuthenticationError):
            await adapter.generate(request, decision)

        # Test 429 Rate limit error
        mock_post.return_value = httpx.Response(status_code=429, text="Rate limit reached")
        with pytest.raises(RateLimitExceededError):
            await adapter.generate(request, decision)

        # Test 400 Bad request
        mock_post.return_value = httpx.Response(status_code=400, text="Invalid model")
        with pytest.raises(InvalidRequestError):
            await adapter.generate(request, decision)
