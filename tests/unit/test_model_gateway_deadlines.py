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


class Stalled(ModelProviderAdapter):
    """Streams `before_stall` chunks, then waits far past any deadline for
    the next one, recording whether that wait was cancelled and whether the
    stream was closed."""

    def __init__(self, before_stall: int):
        self.before_stall = before_stall
        self.cancelled = False
        self.closed = False

    @property
    def name(self) -> str:
        return "stalled"

    def supports_model(self, model: str) -> bool:
        return True

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        raise NotImplementedError

    async def generate_stream(
        self, request: ModelRequest, decision: ModelDecision
    ) -> AsyncIterator[ModelStreamChunk]:
        try:
            for i in range(self.before_stall):
                yield ModelStreamChunk(
                    delta_content=str(i), model=decision.model, provider="stalled"
                )
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            yield ModelStreamChunk(delta_content="late", model=decision.model, provider="stalled")
        finally:
            self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("before_stall", [0, 1])
async def test_a_stalled_stream_is_cancelled_at_the_deadline(before_stall):
    stalled = Stalled(before_stall)
    tracker = BudgetTracker()
    tracker.set_budget("exec_1", BudgetLimits(max_model_calls=5))
    received = []
    started = time.monotonic()

    with pytest.raises(ModelTimeoutError) as caught:
        async for chunk in gateway(stalled, tracker=tracker).generate_stream(
            request("stalled", deadline=0.2)
        ):
            received.append(chunk)

    # Cancelled at the deadline: no next chunk was needed to notice it.
    assert time.monotonic() - started < 0.6
    assert len(received) == before_stall
    assert stalled.cancelled and stalled.closed
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    # Settled: nothing left reserved; charged only if the provider delivered.
    assert tracker.get_reserved("exec_1").model_calls == 0
    assert tracker.get_usage("exec_1").model_calls == before_stall


@pytest.mark.asyncio
async def test_an_adapters_own_timeout_error_is_an_adapter_failure_not_the_deadline():
    class RaisesTimeout(Stalled):
        async def generate_stream(self, request, decision):
            raise TimeoutError("adapter internals")
            yield  # pragma: no cover - makes this an async generator

    with pytest.raises(Exception) as caught:
        async for _ in gateway(RaisesTimeout(0)).generate_stream(request("stalled", deadline=30.0)):
            pass

    assert not isinstance(caught.value, ModelTimeoutError)
    assert "adapter internals" not in str(caught.value)


# --- A provider slow to stop doesn't hold the request ------------------------------
# Cancellation is cooperative: a provider can catch it and keep working (a slow
# cleanup). The request still ends on time; the provider's cleanup gets
# PROVIDER_CLEANUP_SECONDS, then is cancelled again and left behind.


@pytest.fixture
def quick_cleanup(monkeypatch):
    import app.core.models.gateway as gateway_module

    monkeypatch.setattr(gateway_module, "PROVIDER_CLEANUP_SECONDS", 0.1)


class SlowToStop(Stalled):
    """Stalls, and when cancelled or closed spends a minute cleaning up,
    recording whether that cleanup was itself cancelled."""

    def __init__(self, before_stall: int = 0):
        super().__init__(before_stall)
        self.cleanup_cancelled = False

    async def _slow_cleanup(self) -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cleanup_cancelled = True
            raise

    async def generate(self, request, decision):
        try:
            await asyncio.sleep(60)
        finally:
            await self._slow_cleanup()
        raise AssertionError("unreachable")  # pragma: no cover

    async def generate_stream(self, request, decision):
        try:
            for i in range(self.before_stall):
                yield ModelStreamChunk(
                    delta_content=str(i), model=decision.model, provider="stalled"
                )
            await asyncio.sleep(60)
        finally:
            await self._slow_cleanup()


def budgeted() -> BudgetTracker:
    tracker = BudgetTracker()
    tracker.set_budget("exec_1", BudgetLimits(max_model_calls=5))
    return tracker


async def eventually(check) -> bool:
    """`check` holds within half a second: work left behind gets its second
    cancel on a later turn of the event loop."""
    for _ in range(50):
        if check():
            return True
        await asyncio.sleep(0.01)
    return False


async def settled(tracker: BudgetTracker) -> bool:
    return await eventually(lambda: tracker.get_reserved("exec_1").model_calls == 0)


@pytest.mark.asyncio
async def test_a_call_slow_to_stop_does_not_hold_a_timed_out_request(quick_cleanup):
    slow, tracker = SlowToStop(), budgeted()
    started = time.monotonic()

    with pytest.raises(ModelTimeoutError) as caught:
        await gateway(slow, tracker=tracker).generate(request("stalled", deadline=0.2))

    # The deadline (0.2 s) plus the cleanup bound (0.1 s), not the minute.
    assert time.monotonic() - started < 0.8
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert await settled(tracker)
    assert await eventually(lambda: slow.cleanup_cancelled)


@pytest.mark.asyncio
async def test_a_call_slow_to_stop_does_not_hold_a_cancelled_caller(quick_cleanup):
    slow, tracker = SlowToStop(), budgeted()
    call = asyncio.ensure_future(
        gateway(slow, tracker=tracker).generate(request("stalled", deadline=30.0))
    )
    await asyncio.sleep(0.05)
    started = time.monotonic()

    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    assert time.monotonic() - started < 0.6
    assert await settled(tracker)
    assert await eventually(lambda: slow.cleanup_cancelled)


@pytest.mark.asyncio
async def test_a_stream_slow_to_stop_does_not_hold_a_timed_out_request(quick_cleanup):
    slow, tracker = SlowToStop(before_stall=0), budgeted()
    started = time.monotonic()

    with pytest.raises(ModelTimeoutError):
        async for _ in gateway(slow, tracker=tracker).generate_stream(
            request("stalled", deadline=0.2)
        ):
            pass

    assert time.monotonic() - started < 0.8
    assert await settled(tracker)
    assert await eventually(lambda: slow.cleanup_cancelled)


@pytest.mark.asyncio
async def test_a_stream_slow_to_close_does_not_hold_a_caller_that_stopped_reading(
    quick_cleanup,
):
    slow, tracker = SlowToStop(before_stall=1), budgeted()
    stream = gateway(slow, tracker=tracker).generate_stream(request("stalled", deadline=30.0))
    await anext(stream)
    started = time.monotonic()

    await stream.aclose()

    assert time.monotonic() - started < 0.6
    assert await settled(tracker)
    # It delivered a chunk, so it is charged its estimate.
    assert tracker.get_usage("exec_1").model_calls == 1
    assert await eventually(lambda: slow.cleanup_cancelled)
