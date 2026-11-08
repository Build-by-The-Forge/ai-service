import asyncio
import logging
from collections.abc import AsyncIterator, Iterable

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
    CircuitBreakerOpenError,
    FallbackExhaustedError,
    ModelGatewayError,
    NoEligibleProviderError,
    RateLimitExceededError,
)
from app.core.models.provider import ModelProviderAdapter
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
        # Registering a provider never makes it eligible: only approved ones
        # are called (MODEL_PROVIDERS_APPROVED, G-33). Empty fails closed.
        self._approved: frozenset[str] = (
            frozenset(approved_providers)
            if approved_providers is not None
            else settings.model_providers_approved
        )
        self._providers: dict[str, ModelProviderAdapter] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self.budget_tracker = budget_tracker or BudgetTracker()
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter()

        # Register default providers
        if providers:
            for p in providers:
                self.register_provider(p)
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

        # 3. Provider Routing & Fallback Loop, over eligible providers only
        attempted_providers: list[str] = []
        last_error: Exception | None = None

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
            for attempt in range(1, decision.max_retries + 1):
                # Reserve before calling, so concurrent calls in this scope
                # can't each pass a check the other is about to exhaust.
                tokens, cost = _estimate(request, decision, adapter)
                reservation = self.budget_tracker.reserve(scope_key, tokens, cost)
                try:
                    response = await adapter.generate(request, decision)
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
                        decision.max_retries,
                        type(exc).__name__,
                        exc.status_code,
                        is_retryable,
                    )

                    if attempt < decision.max_retries and is_retryable:
                        backoff = ModelRetryPolicy.compute_backoff(attempt)
                        await asyncio.sleep(backoff)
                        continue
                    cb.record_failure()
                    break  # Fall through to next provider in fallback list

                self.budget_tracker.reconcile(reservation, response.usage)
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

        cb = self.get_circuit_breaker(decision.provider)
        if not cb.can_execute():
            raise CircuitBreakerOpenError(
                f"Circuit breaker is OPEN for {decision.provider}", provider=decision.provider
            )

        # 2. Budget: reserved before the stream opens. A stream reports no
        # usage, so a stream that ran (fully or until closed) is charged its
        # estimate; one that failed is released.
        tokens, cost = _estimate(request, decision, adapter)
        reservation = self.budget_tracker.reserve(scope_key, tokens, cost)
        failed = False
        try:
            async for chunk in adapter.generate_stream(request, decision):
                yield chunk
            cb.record_success()

        except asyncio.CancelledError:
            logger.info(
                "Streaming generation cancelled by client for execution %s",
                request.context.execution_id,
            )
            raise
        except Exception as raw:
            failed = True
            self.budget_tracker.release(reservation)
            cb.record_failure()
            exc = _sanitized(raw, decision.provider)
            logger.error(
                "Streaming error on %s: %s status=%s",
                decision.provider,
                type(exc).__name__,
                exc.status_code,
            )
            if exc is raw:
                raise
            raise exc from None
        finally:
            if not failed:
                self.budget_tracker.reconcile(
                    reservation, ModelUsage(total_tokens=tokens, estimated_cost_usd=cost)
                )


def _estimate(
    request: ModelRequest, decision: ModelDecision, adapter: ModelProviderAdapter
) -> tuple[int, float]:
    """The most one call could use: its prompt (about four characters a
    token) plus the completion it is allowed, and what that would cost."""
    prompt_tokens = sum(len(m.content or "") for m in request.messages) // 4 + 1
    tokens = prompt_tokens + decision.max_tokens
    return tokens, adapter.estimate_cost(decision.model, prompt_tokens, decision.max_tokens)


def _sanitized(exc: Exception, provider: str) -> ModelGatewayError:
    """A gateway error as the adapter built it, or, for anything else (an
    adapter bug), a fixed message naming only its type: the original text
    may hold provider output or the prompt, so it isn't carried."""
    if isinstance(exc, ModelGatewayError):
        return exc
    return ModelGatewayError(
        f"Provider {provider} failed unexpectedly ({type(exc).__name__})",
        provider=provider,
        retryable=False,
    )


_default_gateway: ModelGateway | None = None


def get_model_gateway() -> ModelGateway:
    global _default_gateway
    if _default_gateway is None:
        _default_gateway = ModelGateway()
    return _default_gateway


def set_model_gateway(gateway: ModelGateway) -> None:
    global _default_gateway
    _default_gateway = gateway
