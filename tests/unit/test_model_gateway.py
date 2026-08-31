import pytest
from app.core.models.budget import BudgetLimits, BudgetTracker
from app.core.models.contracts import (
    ModelInvocationContext,
    ModelMessage,
    ModelPolicy,
    ModelRequest,
    ModelRole,
)
from app.core.models.errors import (
    AuthenticationError,
    BudgetExceededError,
    FallbackExhaustedError,
    ProviderUnavailableError,
    RateLimitExceededError,
)
from app.core.models.gateway import ModelGateway
from app.core.models.providers.mock import MockProviderAdapter
from app.core.models.resilience import TokenBucketRateLimiter


@pytest.fixture
def sample_context():
    return ModelInvocationContext(
        tenant_id="school_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
    )


@pytest.mark.asyncio
async def test_gateway_successful_generation(sample_context):
    mock_provider = MockProviderAdapter(name="mock", default_response="Gateway response")
    gateway = ModelGateway(providers=[mock_provider])

    request = ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="Hello")],
        context=sample_context,
        policy=ModelPolicy(preferred_provider="mock"),
    )

    response = await gateway.generate(request)
    assert response.content == "Gateway response"
    assert response.provider == "mock"
    assert response.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_gateway_retries_transient_error(sample_context):
    mock_provider = MockProviderAdapter(name="mock")
    # Simulate a transient error initially
    mock_provider.set_simulated_error(
        ProviderUnavailableError("503 Service Unavailable", provider="mock")
    )
    gateway = ModelGateway(providers=[mock_provider])

    request = ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="Hello")],
        context=sample_context,
        policy=ModelPolicy(preferred_provider="mock", max_retries=2),
    )

    # In a real retry loop, it tries max_retries times and then raises
    with pytest.raises(ProviderUnavailableError):
        await gateway.generate(request)

    # Provider should have been called max_retries times
    assert len(mock_provider.call_history) == 2


@pytest.mark.asyncio
async def test_gateway_does_not_retry_non_retryable_error(sample_context):
    mock_provider = MockProviderAdapter(name="mock")
    # 401 is non-retryable
    mock_provider.set_simulated_error(AuthenticationError("Invalid API key", provider="mock"))
    gateway = ModelGateway(providers=[mock_provider])

    request = ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="Hello")],
        context=sample_context,
        policy=ModelPolicy(preferred_provider="mock", max_retries=3),
    )

    with pytest.raises(AuthenticationError):
        await gateway.generate(request)

    # Should only be called once (no retries for auth errors)
    assert len(mock_provider.call_history) == 1


@pytest.mark.asyncio
async def test_gateway_fallbacks_to_secondary_provider(sample_context):
    primary = MockProviderAdapter(name="primary")
    primary.set_simulated_error(ProviderUnavailableError("Primary down", provider="primary"))

    fallback = MockProviderAdapter(name="secondary", default_response="Secondary response")

    gateway = ModelGateway(providers=[primary, fallback])

    request = ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="Hello")],
        context=sample_context,
        policy=ModelPolicy(
            preferred_provider="primary",
            fallback_providers=["secondary"],
            max_retries=1,
        ),
    )

    response = await gateway.generate(request)
    assert response.content == "Secondary response"
    assert response.provider == "secondary"


@pytest.mark.asyncio
async def test_gateway_fails_when_all_fallbacks_exhausted(sample_context):
    primary = MockProviderAdapter(name="p1")
    primary.set_simulated_error(ProviderUnavailableError("P1 down", provider="p1"))

    fallback = MockProviderAdapter(name="p2")
    fallback.set_simulated_error(ProviderUnavailableError("P2 down", provider="p2"))

    gateway = ModelGateway(providers=[primary, fallback])

    request = ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="Hello")],
        context=sample_context,
        policy=ModelPolicy(
            preferred_provider="p1",
            fallback_providers=["p2"],
            max_retries=1,
        ),
    )

    with pytest.raises(FallbackExhaustedError) as exc_info:
        await gateway.generate(request)
    assert exc_info.value.attempted_providers == ["p1", "p2"]


@pytest.mark.asyncio
async def test_gateway_rejects_when_budget_exceeded(sample_context):
    mock_provider = MockProviderAdapter(name="mock")
    budget_tracker = BudgetTracker()
    budget_tracker.set_budget(sample_context.execution_id, BudgetLimits(max_model_calls=1))

    gateway = ModelGateway(providers=[mock_provider], budget_tracker=budget_tracker)

    request = ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="Call 1")],
        context=sample_context,
        policy=ModelPolicy(preferred_provider="mock"),
    )

    # 1st call succeeds
    await gateway.generate(request)

    # 2nd call rejected by budget
    with pytest.raises(BudgetExceededError):
        await gateway.generate(request)


@pytest.mark.asyncio
async def test_gateway_rejects_when_rate_limited(sample_context):
    mock_provider = MockProviderAdapter(name="mock")
    rate_limiter = TokenBucketRateLimiter(rate_per_second=1.0, burst_capacity=1)

    gateway = ModelGateway(providers=[mock_provider], rate_limiter=rate_limiter)

    request = ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="Call 1")],
        context=sample_context,
        policy=ModelPolicy(preferred_provider="mock"),
    )

    # 1st call consumes the 1 token
    await gateway.generate(request)

    # 2nd call rejected by rate limiter
    with pytest.raises(RateLimitExceededError):
        await gateway.generate(request)
