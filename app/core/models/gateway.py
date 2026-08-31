import asyncio
import logging
from collections.abc import AsyncIterator

from app.core.models.budget import BudgetTracker
from app.core.models.contracts import (
    ModelDecision,
    ModelPolicy,
    ModelRequest,
    ModelResponse,
    ModelStreamChunk,
)
from app.core.models.errors import (
    CircuitBreakerOpenError,
    FallbackExhaustedError,
    ModelGatewayError,
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
    ):
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

        provider_name = policy.preferred_provider
        if provider_name not in self._providers and self._providers:
            provider_name = next(iter(self._providers.keys()))

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

    async def generate(self, request: ModelRequest) -> ModelResponse:
        decision = self.resolve_policy(request)
        scope_key = request.context.execution_id or request.context.tenant_id

        # 1. Budget Governance Check
        self.budget_tracker.check_pre_invocation(scope_key, requested_tokens=decision.max_tokens)

        # 2. Rate Limiting Check
        if not self.rate_limiter.acquire(request.context.tenant_id):
            raise RateLimitExceededError(
                f"Rate limit exceeded for tenant {request.context.tenant_id}",
                provider=decision.provider,
            )

        # 3. Provider Routing & Fallback Loop
        attempted_providers: list[str] = []
        candidate_providers = [decision.provider] + [
            p for p in decision.fallback_providers if p != decision.provider
        ]
        for p in self._providers.keys():
            if p not in candidate_providers:
                candidate_providers.append(p)

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
                try:
                    response = await adapter.generate(request, decision)
                    cb.record_success()
                    self.budget_tracker.record_usage(scope_key, response.usage)
                    return response

                except Exception as exc:
                    last_error = exc
                    is_retryable = ModelRetryPolicy.is_retryable(exc)

                    logger.warning(
                        "Model invocation error on %s attempt %d/%d: %s (retryable=%s)",
                        provider_name,
                        attempt,
                        decision.max_retries,
                        str(exc),
                        is_retryable,
                    )

                    if attempt < decision.max_retries and is_retryable:
                        backoff = ModelRetryPolicy.compute_backoff(attempt)
                        await asyncio.sleep(backoff)
                    else:
                        cb.record_failure()
                        break  # Fall through to next provider in fallback list

        # If we got here, all providers failed
        if len(attempted_providers) > 1:
            raise FallbackExhaustedError(
                f"All candidate providers ({attempted_providers}) failed. Last error: {last_error!s}",
                attempted_providers=attempted_providers,
            )
        elif last_error:
            raise last_error
        else:
            raise ModelGatewayError(f"Failed to execute model request on {decision.provider}")

    async def generate_stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        decision = self.resolve_policy(request)
        scope_key = request.context.execution_id or request.context.tenant_id

        # 1. Budget Governance Check
        self.budget_tracker.check_pre_invocation(scope_key, requested_tokens=decision.max_tokens)

        # 2. Rate Limiting Check
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
        except Exception as exc:
            cb.record_failure()
            logger.error("Streaming error on %s: %s", decision.provider, str(exc))
            raise


_default_gateway: ModelGateway | None = None


def get_model_gateway() -> ModelGateway:
    global _default_gateway
    if _default_gateway is None:
        _default_gateway = ModelGateway()
    return _default_gateway


def set_model_gateway(gateway: ModelGateway) -> None:
    global _default_gateway
    _default_gateway = gateway
