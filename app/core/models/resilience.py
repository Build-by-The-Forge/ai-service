import random
import time
from enum import Enum

from app.core.models.errors import (
    AuthenticationError,
    BudgetExceededError,
    CircuitBreakerOpenError,
    InvalidRequestError,
    ModelGatewayError,
    ModelTimeoutError,
    ProviderUnavailableError,
    RateLimitExceededError,
)


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """
    Circuit Breaker state machine for individual model providers.
    Transitions:
      - CLOSED: Invocations pass through. Failures increment counter.
      - OPEN: Failures exceeded threshold. Invocations rejected during cooldown.
      - HALF_OPEN: Cooldown expired. Probe invocations allowed to test provider recovery.
    """

    def __init__(
        self,
        provider_name: str,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        half_open_success_threshold: int = 2,
    ):
        self.provider_name = provider_name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_success_threshold = half_open_success_threshold

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if time.time() - self._opened_at >= self.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
                self._consecutive_successes = 0
        return self._state

    def can_execute(self) -> bool:
        current_state = self.state
        if current_state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
            return True
        return False

    def record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._consecutive_successes += 1
            if self._consecutive_successes >= self.half_open_success_threshold:
                self._state = CircuitState.CLOSED
                self._consecutive_failures = 0
                self._opened_at = None
        elif self._state == CircuitState.CLOSED:
            self._consecutive_failures = 0

    def record_failure(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = time.time()
            self._consecutive_successes = 0
        elif self._state == CircuitState.CLOSED:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.time()


class TokenBucketRateLimiter:
    """
    In-memory rate limiter using the Token Bucket algorithm.
    Enforces per-tenant and per-provider concurrency / throughput limits.
    """

    def __init__(self, rate_per_second: float = 20.0, burst_capacity: int = 50):
        self.rate_per_second = rate_per_second
        self.burst_capacity = burst_capacity
        self._buckets: dict[str, float] = {}
        self._last_refill: dict[str, float] = {}

    def acquire(self, key: str, tokens: int = 1) -> bool:
        now = time.time()
        last_refill = self._last_refill.get(key, now)
        current_tokens = self._buckets.get(key, float(self.burst_capacity))

        # Refill tokens
        elapsed = now - last_refill
        refilled_tokens = min(
            float(self.burst_capacity), current_tokens + elapsed * self.rate_per_second
        )

        if refilled_tokens >= tokens:
            self._buckets[key] = refilled_tokens - tokens
            self._last_refill[key] = now
            return True
        else:
            self._buckets[key] = refilled_tokens
            self._last_refill[key] = now
            return False


class ModelRetryPolicy:
    """
    Determines retry eligibility and backoff durations for model errors.
    """

    @staticmethod
    def is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (ModelTimeoutError, ProviderUnavailableError, RateLimitExceededError)):
            return True
        if isinstance(
            exc,
            (
                AuthenticationError,
                InvalidRequestError,
                BudgetExceededError,
                CircuitBreakerOpenError,
            ),
        ):
            return False
        if isinstance(exc, ModelGatewayError):
            return exc.retryable
        return False

    @staticmethod
    def compute_backoff(
        attempt: int,
        initial_backoff: float = 0.5,
        max_backoff: float = 5.0,
        jitter: bool = True,
    ) -> float:
        backoff = min(initial_backoff * (2 ** (attempt - 1)), max_backoff)
        if jitter:
            backoff += random.uniform(0, 0.2 * backoff)
        return backoff
