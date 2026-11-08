"""AI-01 PR 1, F5: one deadline across retries and fallbacks.

The whole request finishes before its deadline, the backend's 60 s call
deadline being the outer bound. When the deadline passes, the call in flight
is cancelled, not abandoned to finish (and be paid for) after the caller has
given up.
"""

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from app.core.models.budget import BudgetLimits, BudgetTracker
from app.core.models.contracts import (
    ModelDecision,
    ModelInvocationContext,
    ModelMessage,
    ModelPolicy,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelStreamChunk,
    ModelUsage,
)
from app.core.models.errors import ModelTimeoutError, ProviderUnavailableError
from app.core.models.gateway import ModelGateway
from app.core.models.provider import ModelProviderAdapter

CONTEXT = ModelInvocationContext(
    tenant_id="school_1", execution_id="exec_1", task_id="t", correlation_id="c"
)


class Slow(ModelProviderAdapter):
    """Answers after `latency`, recording each call's timeout and whether it
    was cancelled; streams one chunk every `latency`."""

    def __init__(self, name: str, latency: float, error: Exception | None = None):
        self._name = name
        self.latency = latency
        self.error = error
        self.calls = 0
        self.cancelled = 0
        self.timeouts: list[float] = []
        self.stream_closed = False

    @property
    def name(self) -> str:
        return self._name

    def supports_model(self, model: str) -> bool:
        return True

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        self.calls += 1
        self.timeouts.append(decision.timeout_seconds)
        if self.error is not None:
            raise self.error
        try:
            await asyncio.sleep(self.latency)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return ModelResponse(
            content="late",
            model=decision.model,
            provider=self._name,
            role=ModelRole.ASSISTANT,
            finish_reason="stop",
            usage=ModelUsage(total_tokens=1),
            latency_ms=0.0,
            context=request.context,
        )

    async def generate_stream(
        self, request: ModelRequest, decision: ModelDecision
    ) -> AsyncIterator[ModelStreamChunk]:
        try:
            for i in range(100):
                await asyncio.sleep(self.latency)
                yield ModelStreamChunk(
                    delta_content=str(i), model=decision.model, provider=self._name
                )
        finally:
            self.stream_closed = True


def request(
    *providers: str, deadline: float, retries: int = 0, deadline_at: float | None = None
) -> ModelRequest:
    return ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="hi")],
        context=CONTEXT,
        policy=ModelPolicy(
            preferred_provider=providers[0],
            fallback_providers=list(providers[1:]),
            max_retries=retries,
            deadline_seconds=deadline,
        ),
        deadline_at=deadline_at,
    )


def gateway(*adapters: ModelProviderAdapter, tracker: BudgetTracker | None = None) -> ModelGateway:
    return ModelGateway(
        providers=list(adapters),
        approved_providers={a.name for a in adapters},
        budget_tracker=tracker,
    )


@pytest.mark.asyncio
async def test_the_deadline_bounds_retries_and_fallbacks_together():
    a, b = Slow("a", latency=1.0), Slow("b", latency=1.0)
    started = time.monotonic()

    with pytest.raises(ModelTimeoutError):
        await gateway(a, b).generate(request("a", "b", deadline=0.2, retries=3))

    assert time.monotonic() - started < 0.6
    assert a.calls + b.calls == 1


@pytest.mark.asyncio
async def test_the_call_in_flight_is_cancelled_when_the_deadline_passes():
    slow = Slow("slow", latency=5.0)

    with pytest.raises(ModelTimeoutError):
        await gateway(slow).generate(request("slow", deadline=0.1))

    assert slow.cancelled == 1


@pytest.mark.asyncio
async def test_an_earlier_caller_deadline_wins():
    slow = Slow("slow", latency=5.0)
    started = time.monotonic()

    with pytest.raises(ModelTimeoutError):
        await gateway(slow).generate(
            request("slow", deadline=45.0, deadline_at=time.monotonic() + 0.1)
        )

    assert time.monotonic() - started < 0.5


@pytest.mark.asyncio
async def test_no_attempt_is_given_longer_than_the_deadline_has_left():
    quick = Slow("quick", latency=0.0)

    await gateway(quick).generate(request("quick", deadline=2.0))

    assert 0 < quick.timeouts[0] <= 2.0


@pytest.mark.asyncio
async def test_a_retry_whose_backoff_would_outlast_the_deadline_is_not_attempted():
    # The first backoff is at least 0.5 s; with 0.3 s left, retrying is futile.
    failing = Slow(
        "failing", latency=0.0, error=ProviderUnavailableError("down", provider="failing")
    )
    started = time.monotonic()

    with pytest.raises(ProviderUnavailableError):
        await gateway(failing).generate(request("failing", deadline=0.3, retries=5))

    assert failing.calls == 1
    assert time.monotonic() - started < 0.3


@pytest.mark.asyncio
async def test_a_timed_out_call_gives_its_budget_reservation_back():
    tracker = BudgetTracker()
    tracker.set_budget("exec_1", BudgetLimits(max_model_calls=5))

    with pytest.raises(ModelTimeoutError):
        await gateway(Slow("slow", latency=5.0), tracker=tracker).generate(
            request("slow", deadline=0.1)
        )

    assert tracker.get_reserved("exec_1").model_calls == 0


@pytest.mark.asyncio
async def test_a_stream_that_outlasts_the_deadline_is_closed_and_fails():
    streaming = Slow("streaming", latency=0.05)
    received = []

    with pytest.raises(ModelTimeoutError):
        async for chunk in gateway(streaming).generate_stream(request("streaming", deadline=0.2)):
            received.append(chunk)

    assert 0 < len(received) < 100
    assert streaming.stream_closed


def test_the_default_policy_retries_once_inside_the_backends_deadline():
    policy = ModelPolicy()
    assert policy.max_retries == 1
    assert policy.deadline_seconds < 60
