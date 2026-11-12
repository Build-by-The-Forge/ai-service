import asyncio
import dataclasses
import json
import logging
import math
import time
from collections.abc import AsyncIterator, Iterable
from typing import Any, NoReturn, cast

from app.core.config import settings

from app.core.models.budget import BudgetTracker
from app.core.models.contracts import (
    ModelDecision,
    ModelPolicy,
    ModelRequest,
    ModelResponse,
    ModelStreamChunk,
    ModelUsage,
)
from app.core.models.errors import (
    AuthenticationError,
    CircuitBreakerOpenError,
    FallbackExhaustedError,
    InvalidRequestError,
    InvalidStructuredOutputError,
    ModelGatewayError,
    ModelTimeoutError,
    NoEligibleProviderError,
    ProviderUnavailableError,
    RateLimitExceededError,
)
from app.core.models.provider import ModelProviderAdapter
from app.core.models.providers.deterministic import NAME as DETERMINISTIC
from app.core.models.providers.deterministic import DeterministicProviderAdapter
from app.core.models.providers.openai import OpenAIProviderAdapter
from app.core.models.resilience import (
    CircuitBreaker,
    ModelRetryPolicy,
    TokenBucketRateLimiter,
)

logger = logging.getLogger(__name__)


class ModelGateway:
    """
    Centralized Model Control Plane coordinating policy resolution, budget governance,
    rate limiting, circuit breaking, retry execution, and fallback provider routing.
    """

    def __init__(
        self,
        providers: list[ModelProviderAdapter] | None = None,
        budget_tracker: BudgetTracker | None = None,
        rate_limiter: TokenBucketRateLimiter | None = None,
        approved_providers: Iterable[str] | None = None,
    ):
        # Deterministic mode (development and tests only: refused in strict
        # environments) is offline. Its adapter is the only one registered or
        # approved, whatever MODEL_PROVIDERS_APPROVED says, so no request in
        # that mode can reach a network provider.
        offline = not providers and settings.MODEL_PROVIDER == DETERMINISTIC
        # Registering a provider never makes it eligible: only approved ones
        # are called (MODEL_PROVIDERS_APPROVED, G-33). Empty fails closed.
        if approved_providers is not None:
            self._approved: frozenset[str] = frozenset(approved_providers)
        elif offline:
            self._approved = frozenset({DETERMINISTIC})
        else:
            self._approved = settings.model_providers_approved
        self._providers: dict[str, ModelProviderAdapter] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self.budget_tracker = budget_tracker or BudgetTracker()
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter()

        # Register default providers
        if providers:
            for p in providers:
                self.register_provider(p)
        elif offline:
            self.register_provider(DeterministicProviderAdapter())
        else:
            self.register_provider(OpenAIProviderAdapter())

    def register_provider(self, provider: ModelProviderAdapter) -> None:
        self._providers[provider.name] = provider
        if provider.name not in self._circuit_breakers:
            self._circuit_breakers[provider.name] = CircuitBreaker(provider.name)

    def get_provider(self, name: str) -> ModelProviderAdapter | None:
        return self._providers.get(name)

    def get_circuit_breaker(self, provider_name: str) -> CircuitBreaker:
        if provider_name not in self._circuit_breakers:
            self._circuit_breakers[provider_name] = CircuitBreaker(provider_name)
        return self._circuit_breakers[provider_name]

    def resolve_policy(self, request: ModelRequest) -> ModelDecision:
        """
        Resolves a ModelRequest and its declarative ModelPolicy into an immutable ModelDecision.
        """
        policy = request.policy or ModelPolicy()

        # Resolve model name based on policy quality tier if not explicitly specified
        if request.model:
            model_name = request.model
        elif policy.quality == "high":
            model_name = "gpt-4o"
        elif policy.quality == "fast":
            model_name = "gpt-4o-mini"
        else:  # balanced
            model_name = "gpt-4o-mini"

        # The policy's provider, never a substitute for it.
        provider_name = policy.preferred_provider

        max_tokens = request.max_tokens or policy.max_tokens
        timeout_seconds = policy.timeout_seconds
        max_retries = policy.max_retries
        temperature = request.temperature if request.temperature is not None else policy.temperature

        return ModelDecision(
            provider=provider_name,
            model=model_name,
            fallback_providers=policy.fallback_providers,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            max_cost_usd=policy.max_cost_usd,
            temperature=temperature,
        )

    def eligible_providers(self, decision: ModelDecision) -> list[str]:
        """The preferred provider, then its named fallbacks: each only if it is
        approved and registered. Nothing else, however many are registered."""
        named = [decision.provider, *decision.fallback_providers]
        return [p for p in dict.fromkeys(named) if p in self._approved and p in self._providers]

    def _eligible_or_refuse(self, decision: ModelDecision) -> list[str]:
        eligible = self.eligible_providers(decision)
        if not eligible:
            raise NoEligibleProviderError(
                "No approved, registered provider among the policy's providers",
                provider=decision.provider,
            )
        return eligible

    async def generate(self, request: ModelRequest) -> ModelResponse:
        decision = self.resolve_policy(request)
        candidate_providers = self._eligible_or_refuse(decision)
        scope_key = request.context.execution_id or request.context.tenant_id

        # 1. Rate Limiting Check (budgets are reserved per attempt, below)
        if not self.rate_limiter.acquire(request.context.tenant_id):
            raise RateLimitExceededError(
                f"Rate limit exceeded for tenant {request.context.tenant_id}",
                provider=decision.provider,
            )

        # 2. One deadline for every attempt and fallback. When it passes, the
        # call in flight is cancelled (its reservation released) and the
        # request ends with ModelTimeoutError.
        deadline = _deadline(request)
        # The provider work runs as a task, so the request ends at its deadline
        # even if a provider is slow to stop: a provider's cleanup on
        # cancellation gets PROVIDER_CLEANUP_SECONDS, then is left behind.
        routed = asyncio.ensure_future(
            self._route(request, decision, candidate_providers, scope_key, deadline)
        )
        try:
            await asyncio.wait({routed}, timeout=max(0.0, _remaining(deadline)))
        except asyncio.CancelledError:
            await _end_within_bound(routed, cancel=True)
            raise
        if routed.done():
            return routed.result()
        await _end_within_bound(routed, cancel=True)
        raise ModelTimeoutError("Model request deadline exceeded", provider=decision.provider)

    async def _route(
        self,
        request: ModelRequest,
        decision: ModelDecision,
        candidate_providers: list[str],
        scope_key: str,
        deadline: float,
    ) -> ModelResponse:
        # 3. Provider Routing & Fallback Loop, over eligible providers only
        attempted_providers: list[str] = []
        last_error: Exception | None = None
        attempts = decision.max_retries + 1

        for provider_name in candidate_providers:
            attempted_providers.append(provider_name)
            adapter = self.get_provider(provider_name)
            if not adapter:
                logger.warning("Provider adapter %s not registered. Skipping.", provider_name)
                continue

            cb = self.get_circuit_breaker(provider_name)
            if not cb.can_execute():
                logger.warning("Circuit breaker OPEN for provider %s. Skipping.", provider_name)
                last_error = CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN for {provider_name}", provider=provider_name
                )
                continue

            # Retry loop per provider
            for attempt in range(1, attempts + 1):
                # Reserve before calling, so concurrent calls in this scope
                # can't each pass a check the other is about to exhaust.
                tokens, cost = _estimate(request, decision, adapter)
                reservation = self.budget_tracker.reserve(scope_key, tokens, cost)
                # No attempt may outlast the request's deadline.
                attempt_decision = dataclasses.replace(
                    decision,
                    timeout_seconds=max(0.001, min(decision.timeout_seconds, _remaining(deadline))),
                )
                try:
                    response = await adapter.generate(request, attempt_decision)
                except BaseException as raw:
                    self.budget_tracker.release(reservation)
                    if not isinstance(raw, Exception):
                        raise  # cancellation: released, then propagated
                    exc = _sanitized(raw, provider_name)
                    last_error = exc
                    is_retryable = ModelRetryPolicy.is_retryable(exc)

                    # Types and statuses only: no exception text reaches the logs.
                    logger.warning(
                        "Model invocation error on %s attempt %d/%d: %s status=%s (retryable=%s)",
                        provider_name,
                        attempt,
                        attempts,
                        type(exc).__name__,
                        exc.status_code,
                        is_retryable,
                    )

                    backoff = ModelRetryPolicy.compute_backoff(attempt)
                    if attempt < attempts and is_retryable and backoff < _remaining(deadline):
                        await asyncio.sleep(backoff)
                        continue
                    cb.record_failure()
                    break  # Fall through to next provider in fallback list

                # The provider answered, so it is charged, whether or not
                # the answer is usable.
                self.budget_tracker.reconcile(reservation, response.usage)
                if request.structured is not None:
                    parsed = _parse_structured(response)
                    if parsed is None:
                        last_error = InvalidStructuredOutputError(
                            f"Provider {provider_name} did not answer with a JSON object",
                            provider=provider_name,
                        )
                        logger.warning(
                            "Model invocation error on %s attempt %d/%d: %s",
                            provider_name,
                            attempt,
                            attempts,
                            type(last_error).__name__,
                        )
                        cb.record_failure()
                        break  # Not retried: try the next named fallback
                    response = parsed
                cb.record_success()
                return response

        # If we got here, all providers failed
        if len(attempted_providers) > 1:
            raise FallbackExhaustedError(
                f"All candidate providers ({attempted_providers}) failed. "
                f"Last error: {type(last_error).__name__}",
                attempted_providers=attempted_providers,
            )
        elif last_error:
            raise last_error from None
        else:
            raise ModelGatewayError(f"Failed to execute model request on {decision.provider}")

    async def generate_stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        decision = self.resolve_policy(request)
        # A structured answer is checked whole, so it can't be streamed.
        if request.structured is not None:
            raise InvalidRequestError(
                "A structured request can't be streamed", provider=decision.provider
            )
        # Streaming has no fallback: only the preferred provider, if eligible.
        if decision.provider not in self.eligible_providers(decision):
            raise NoEligibleProviderError(
                "The policy's provider is not approved and registered", provider=decision.provider
            )
        scope_key = request.context.execution_id or request.context.tenant_id

        # 1. Rate Limiting Check
        if not self.rate_limiter.acquire(request.context.tenant_id):
            raise RateLimitExceededError(
                f"Rate limit exceeded for tenant {request.context.tenant_id}",
                provider=decision.provider,
            )

        adapter = self.get_provider(decision.provider)
        if not adapter:
            raise ModelGatewayError(f"Provider {decision.provider} not registered")
        deadline = _deadline(request)
        decision = dataclasses.replace(
            decision,
            timeout_seconds=max(0.001, min(decision.timeout_seconds, _remaining(deadline))),
        )

        cb = self.get_circuit_breaker(decision.provider)
        if not cb.can_execute():
            raise CircuitBreakerOpenError(
                f"Circuit breaker is OPEN for {decision.provider}", provider=decision.provider
            )

        # 2. Budget: reserved before the stream opens. A stream reports no
        # usage. Once it has delivered a chunk the provider has done work, so
        # it is charged its estimate however it ends; a stream that delivered
        # nothing is released.
        tokens, cost = _estimate(request, decision, adapter)
        reservation = self.budget_tracker.reserve(scope_key, tokens, cost)
        delivered = False
        # Set when the provider's stream is still running after its bounded
        # cancellation: it can't be closed, only left behind.
        left_running = False
        error: ModelGatewayError | None = None
        chunks = adapter.generate_stream(request, decision)
        try:
            while True:
                # The request's absolute deadline covers every wait for the
                # next chunk: a stalled provider is cancelled at the deadline,
                # not when (or if) its next chunk arrives. The wait runs as a
                # task, so a provider slow to stop can't hold the request.
                pending = asyncio.ensure_future(_next_chunk(chunks))
                try:
                    await asyncio.wait({pending}, timeout=max(0.0, _remaining(deadline)))
                except asyncio.CancelledError:
                    left_running = not await _end_within_bound(pending, cancel=True)
                    raise
                if not pending.done():
                    left_running = not await _end_within_bound(pending, cancel=True)
                    error = ModelTimeoutError(
                        "Model request deadline exceeded", provider=decision.provider
                    )
                    break
                chunk = pending.result()
                if chunk is None:
                    break
                delivered = True
                yield chunk
        except asyncio.CancelledError:
            logger.info(
                "Streaming generation cancelled by client for execution %s",
                request.context.execution_id,
            )
            raise
        except Exception as raw:
            error = _sanitized(raw, decision.provider)
        finally:
            # Close the provider's stream now, ending any work it still has,
            # then settle the reservation, however the stream ended.
            if not left_running:
                await _close_stream(chunks)
            if delivered:
                self.budget_tracker.reconcile(
                    reservation, ModelUsage(total_tokens=tokens, estimated_cost_usd=cost)
                )
            else:
                self.budget_tracker.release(reservation)

        if error is not None:
            cb.record_failure()
            logger.error(
                "Streaming error on %s: %s status=%s",
                decision.provider,
                type(error).__name__,
                error.status_code,
            )
            # Raised outside any handler, so no adapter exception is attached
            # as its cause or context.
            raise error
        cb.record_success()


# How long a provider's work may take to stop once the gateway is done with
# it: cancelled at the deadline, or its stream closed. Its cleanup must not
# hold a request that has already finished or timed out.
PROVIDER_CLEANUP_SECONDS = 2.0


async def _next_chunk(chunks: AsyncIterator[ModelStreamChunk]) -> ModelStreamChunk | None:
    """The stream's next chunk, or None at its end."""
    try:
        return await anext(chunks)
    except StopAsyncIteration:
        return None


async def _aclose(chunks: AsyncIterator[ModelStreamChunk]) -> None:
    aclose = getattr(chunks, "aclose", None)
    if aclose is not None:
        await aclose()


async def _close_stream(chunks: AsyncIterator[ModelStreamChunk]) -> None:
    """Close a provider's stream, waiting at most PROVIDER_CLEANUP_SECONDS.
    A failure to close is logged, never raised: it mustn't replace the
    stream's outcome."""
    await _end_within_bound(asyncio.ensure_future(_aclose(chunks)), cancel=False)


async def _end_within_bound(work: "asyncio.Future[Any]", *, cancel: bool) -> bool:
    """Wait at most PROVIDER_CLEANUP_SECONDS for provider work to end,
    cancelling it first if `cancel`. True if it ended. Work still running after
    that is cancelled (again) and left to finish on its own, its outcome
    consumed then, so the request that started it isn't held."""
    if cancel:
        work.cancel()
    try:
        await asyncio.wait({work}, timeout=PROVIDER_CLEANUP_SECONDS)
    finally:
        if work.done():
            _consume(work)
        else:
            work.cancel()
            work.add_done_callback(_consume)
            logger.warning(
                "Provider work still running %.1f s after the gateway ended it; left behind",
                PROVIDER_CLEANUP_SECONDS,
            )
    return work.done()


def _consume(work: "asyncio.Future[Any]") -> None:
    """Retrieve finished provider work's outcome, so asyncio never reports it
    as unhandled. Only the error's type is logged, never its text."""
    if work.cancelled():
        return
    exc = work.exception()
    if exc is not None:
        logger.warning("Provider cleanup failed: %s", type(exc).__name__)


def _not_json(constant: str) -> NoReturn:
    raise ValueError("NaN and Infinity are not JSON")


def _finite(number: str) -> float:
    value = float(number)
    if not math.isfinite(value):  # such as 1e400, which overflows to infinity
        raise ValueError("a number too large for a float")
    return value


def _parse_structured(response: ModelResponse) -> ModelResponse | None:
    """The response with its JSON object parsed into `structured`, or None if
    its content isn't one standard JSON object. Python's parser also accepts
    NaN, Infinity and -Infinity, and turns 1e400 into infinity: none of them
    is JSON, and each would be written back out as invalid JSON, so each is
    refused, as is nesting too deep to parse."""
    try:
        value = json.loads(response.content, parse_constant=_not_json, parse_float=_finite)
    except (ValueError, RecursionError):
        return None
    if not isinstance(value, dict):
        return None
    return dataclasses.replace(response, structured=value)


def _deadline(request: ModelRequest) -> float:
    """The request's deadline on the time.monotonic() clock: the policy's,
    or the caller's if that comes first."""
    policy = request.policy or ModelPolicy()
    deadline = time.monotonic() + policy.deadline_seconds
    if request.deadline_at is not None:
        deadline = min(deadline, request.deadline_at)
    return deadline


def _remaining(deadline: float) -> float:
    return deadline - time.monotonic()


def _estimate(
    request: ModelRequest, decision: ModelDecision, adapter: ModelProviderAdapter
) -> tuple[int, float]:
    """The most one call could use: its prompt (about four characters a
    token) plus the completion it is allowed, and what that would cost."""
    prompt_tokens = sum(len(m.content or "") for m in request.messages) // 4 + 1
    tokens = prompt_tokens + decision.max_tokens
    return tokens, adapter.estimate_cost(decision.model, prompt_tokens, decision.max_tokens)


# The error types an adapter may raise. An adapter's own subclass becomes the
# nearest of these, so no extra state it carries survives.
_ADAPTER_ERROR_TYPES: tuple[type[ModelGatewayError], ...] = (
    ProviderUnavailableError,
    ModelTimeoutError,
    RateLimitExceededError,
    AuthenticationError,
    InvalidRequestError,
    CircuitBreakerOpenError,
    ModelGatewayError,
)


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _sanitized(exc: BaseException, provider: str) -> ModelGatewayError:
    """Every adapter error is rebuilt here, at the boundary. Nothing it says
    is trusted, not even from a ModelGatewayError: an adapter may have put a
    passage or a provider body in its message, its provider field or its
    cause. The rebuilt error keeps its class (the nearest adapter error type),
    so retry classification and callers' handling don't change, and carries
    only safe metadata: the gateway's own name for the provider, an integer
    HTTP status, the retry flag and a numeric retry-after. Its message is
    fixed text, and it has no cause."""
    if not isinstance(exc, ModelGatewayError):
        return ModelGatewayError(
            f"Provider {provider} failed unexpectedly ({type(exc).__name__})",
            provider=provider,
            retryable=False,
        )
    cls = cast(
        type[ModelGatewayError],
        next(c for c in type(exc).__mro__ if c in _ADAPTER_ERROR_TYPES),
    )
    status = _int_or_none(exc.status_code)
    detail = f"{cls.__name__}, HTTP {status}" if status is not None else cls.__name__
    safe = cls.__new__(cls)
    ModelGatewayError.__init__(
        safe,
        f"Provider {provider} failed ({detail})",
        provider=provider,
        retryable=bool(exc.retryable),
    )
    safe.status_code = status
    if isinstance(safe, RateLimitExceededError):
        retry_after = getattr(exc, "retry_after", None)
        safe.retry_after = (
            float(retry_after)
            if isinstance(retry_after, int | float) and not isinstance(retry_after, bool)
            else None
        )
    return safe


_default_gateway: ModelGateway | None = None


def get_model_gateway() -> ModelGateway:
    global _default_gateway
    if _default_gateway is None:
        _default_gateway = ModelGateway()
    return _default_gateway


def set_model_gateway(gateway: ModelGateway) -> None:
    global _default_gateway
    _default_gateway = gateway
