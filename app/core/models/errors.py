class ModelGatewayError(Exception):
    """Base exception for all Model Gateway errors."""

    def __init__(self, message: str, provider: str | None = None, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.retryable = retryable


class ProviderUnavailableError(ModelGatewayError):
    """Raised when a model provider is unreachable, returns 5xx, or network connection fails."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider=provider, retryable=True)


class ModelTimeoutError(ModelGatewayError):
    """Raised when a model provider invocation exceeds its deadline or timeout."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider=provider, retryable=True)


class RateLimitExceededError(ModelGatewayError):
    """Raised when a model provider returns HTTP 429 or local rate limit is exceeded."""

    def __init__(self, message: str, provider: str | None = None, retry_after: float | None = None):
        super().__init__(message, provider=provider, retryable=True)
        self.retry_after = retry_after


class CircuitBreakerOpenError(ModelGatewayError):
    """Raised when invocations are rejected because the provider's circuit breaker is in OPEN state."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider=provider, retryable=False)


class BudgetExceededError(ModelGatewayError):
    """Raised when a model invocation violates token, cost, or call count budgets."""

    def __init__(self, message: str, budget_type: str, limit: float, current: float):
        super().__init__(
            f"Budget exceeded for {budget_type}: limit={limit}, current={current}", retryable=False
        )
        self.budget_type = budget_type
        self.limit = limit
        self.current = current


class AuthenticationError(ModelGatewayError):
    """Raised when provider credentials (e.g. API Key) are missing or rejected (HTTP 401/403)."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider=provider, retryable=False)


class InvalidRequestError(ModelGatewayError):
    """Raised when the model request is malformed (HTTP 400/422)."""

    def __init__(self, message: str, provider: str | None = None):
        super().__init__(message, provider=provider, retryable=False)


class FallbackExhaustedError(ModelGatewayError):
    """Raised when all primary and fallback providers have failed for an invocation."""

    def __init__(self, message: str, attempted_providers: list[str]):
        super().__init__(message, retryable=False)
        self.attempted_providers = attempted_providers
