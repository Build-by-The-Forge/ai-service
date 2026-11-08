"""AI-01 PR 1, F4: budgets are reserved before a call and reconciled after it.

Concurrent calls in one scope see each other's reservations, so together
they can't overspend it. Failed and cancelled calls give their reservation
back. Scopes without limits aren't tracked, so nothing accumulates.
"""

import asyncio

import pytest

from app.core.models.budget import BudgetLimits, BudgetTracker
from app.core.models.contracts import (
    ModelInvocationContext,
    ModelMessage,
    ModelPolicy,
    ModelRequest,
    ModelRole,
    ModelUsage,
)
from app.core.models.errors import BudgetExceededError, ProviderUnavailableError
from app.core.models.gateway import ModelGateway
from app.core.models.providers.mock import MockProviderAdapter

SCOPE = "exec_1"


def request(execution_id: str = SCOPE, max_tokens: int = 50) -> ModelRequest:
    return ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="hi")],
        context=ModelInvocationContext(
            tenant_id="school_1", execution_id=execution_id, task_id="t", correlation_id="c"
        ),
        policy=ModelPolicy(preferred_provider="mock", max_retries=1),
        max_tokens=max_tokens,
    )


def gateway_with(provider: MockProviderAdapter, tracker: BudgetTracker) -> ModelGateway:
    return ModelGateway(providers=[provider], approved_providers={"mock"}, budget_tracker=tracker)


@pytest.mark.asyncio
async def test_concurrent_calls_cannot_both_pass_a_call_limit():
    provider = MockProviderAdapter(name="mock", latency_seconds=0.05)
    tracker = BudgetTracker()
    tracker.set_budget(SCOPE, BudgetLimits(max_model_calls=1))
    gateway = gateway_with(provider, tracker)

    results = await asyncio.gather(
        gateway.generate(request()), gateway.generate(request()), return_exceptions=True
    )

    assert sum(isinstance(r, BudgetExceededError) for r in results) == 1
    assert len(provider.call_history) == 1


@pytest.mark.asyncio
async def test_concurrent_calls_cannot_both_pass_a_token_limit():
    # Each may use 51 tokens (a 1-token prompt and 50 allowed); together
    # they could exceed 100, so the second is refused before it is made.
    provider = MockProviderAdapter(name="mock", latency_seconds=0.05)
    tracker = BudgetTracker()
    tracker.set_budget(SCOPE, BudgetLimits(max_tokens=100))
    gateway = gateway_with(provider, tracker)

    results = await asyncio.gather(
        gateway.generate(request()), gateway.generate(request()), return_exceptions=True
    )

    assert sum(isinstance(r, BudgetExceededError) for r in results) == 1
    assert len(provider.call_history) == 1


@pytest.mark.asyncio
async def test_a_call_is_charged_what_it_used_not_what_it_reserved():
    usage = ModelUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    provider = MockProviderAdapter(name="mock", token_usage=usage)
    tracker = BudgetTracker()
    tracker.set_budget(SCOPE, BudgetLimits(max_tokens=1000))
    gateway = gateway_with(provider, tracker)

    for _ in range(3):
        await gateway.generate(request())

    assert tracker.get_usage(SCOPE).total_tokens == 90
    assert tracker.get_usage(SCOPE).model_calls == 3
    assert tracker.get_reserved(SCOPE).total_tokens == 0


@pytest.mark.asyncio
async def test_a_failed_call_gives_its_reservation_back():
    provider = MockProviderAdapter(name="mock")
    provider.set_simulated_error(ProviderUnavailableError("down", provider="mock"))
    tracker = BudgetTracker()
    tracker.set_budget(SCOPE, BudgetLimits(max_model_calls=1))
    gateway = gateway_with(provider, tracker)

    with pytest.raises(ProviderUnavailableError):
        await gateway.generate(request())
    assert tracker.get_reserved(SCOPE).model_calls == 0

    provider.set_simulated_error(None)
    assert (await gateway.generate(request())).provider == "mock"


@pytest.mark.asyncio
async def test_a_cancelled_call_gives_its_reservation_back():
    provider = MockProviderAdapter(name="mock", latency_seconds=5.0)
    tracker = BudgetTracker()
    tracker.set_budget(SCOPE, BudgetLimits(max_model_calls=1))
    gateway = gateway_with(provider, tracker)

    task = asyncio.create_task(gateway.generate(request()))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert tracker.get_reserved(SCOPE).model_calls == 0


class Priced(MockProviderAdapter):
    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        return 1.0


@pytest.mark.asyncio
async def test_an_estimated_cost_over_the_limit_is_refused_before_the_call():
    provider = Priced(name="mock")
    tracker = BudgetTracker()
    tracker.set_budget(SCOPE, BudgetLimits(max_cost_usd=0.5))

    with pytest.raises(BudgetExceededError) as caught:
        await gateway_with(provider, tracker).generate(request())

    assert caught.value.budget_type == "max_cost_usd"
    assert provider.call_history == []


@pytest.mark.asyncio
async def test_scopes_without_limits_leave_nothing_behind():
    tracker = BudgetTracker()
    gateway = gateway_with(MockProviderAdapter(name="mock"), tracker)

    for i in range(50):
        await gateway.generate(request(execution_id=f"exec_{i}"))

    assert tracker.tracked_scope_count == 0


@pytest.mark.asyncio
async def test_a_request_scope_applies_its_limits_and_is_dropped_when_it_ends():
    tracker = BudgetTracker()
    gateway = gateway_with(MockProviderAdapter(name="mock"), tracker)

    with tracker.scope(SCOPE, BudgetLimits(max_model_calls=1)):
        await gateway.generate(request())
        with pytest.raises(BudgetExceededError):
            await gateway.generate(request())

    assert tracker.tracked_scope_count == 0


@pytest.mark.asyncio
async def test_a_stream_is_reserved_and_charged_its_estimate():
    tracker = BudgetTracker()
    tracker.set_budget(SCOPE, BudgetLimits(max_model_calls=1))
    gateway = gateway_with(MockProviderAdapter(name="mock"), tracker)

    async for _ in gateway.generate_stream(request()):
        pass

    assert tracker.get_usage(SCOPE).model_calls == 1
    assert tracker.get_reserved(SCOPE).model_calls == 0
    with pytest.raises(BudgetExceededError):
        async for _ in gateway.generate_stream(request()):
            pass


def test_the_openai_adapter_estimates_cost_from_its_pricing():
    from app.core.models.providers.openai import OpenAIProviderAdapter, calculate_openai_cost

    adapter = OpenAIProviderAdapter(api_key="sk-test")
    assert adapter.estimate_cost("gpt-4o", 1000, 1000) == calculate_openai_cost(
        "gpt-4o", 1000, 1000
    )
