from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from app.core.models.contracts import ModelUsage
from app.core.models.errors import BudgetExceededError


@dataclass
class BudgetLimits:
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_model_calls: int | None = None


@dataclass
class ExecutionBudgetUsage:
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    model_calls: int = 0


@dataclass(frozen=True)
class Reservation:
    """What one call holds against its scope's limits until it settles."""

    scope_key: str
    tokens: int
    cost_usd: float


class BudgetTracker:
    """
    Enforces per-tenant and per-execution token, cost, and call-count budgets.
    Prevents uncontrolled reasoning loops and runaway provider charges.

    A call reserves its estimated usage before it is made and reconciles to
    its actual usage after (or releases the reservation if it fails), so
    concurrent calls in one scope see each other and can't overspend it.
    Only scopes with limits are tracked, and end_scope drops them: nothing
    accumulates for requests that were never limited.
    """

    def __init__(self):
        self._limits: dict[str, BudgetLimits] = {}
        self._usage: dict[str, ExecutionBudgetUsage] = {}
        self._reserved: dict[str, ExecutionBudgetUsage] = {}

    def set_budget(self, scope_key: str, limits: BudgetLimits) -> None:
        self._limits[scope_key] = limits

    @contextmanager
    def scope(self, scope_key: str, limits: BudgetLimits) -> Iterator[None]:
        """Limits for one request, dropped with its usage when it ends."""
        self.set_budget(scope_key, limits)
        try:
            yield
        finally:
            self.end_scope(scope_key)

    def end_scope(self, scope_key: str) -> None:
        self._limits.pop(scope_key, None)
        self._usage.pop(scope_key, None)
        self._reserved.pop(scope_key, None)

    @property
    def tracked_scope_count(self) -> int:
        return len(self._limits.keys() | self._usage.keys() | self._reserved.keys())

    def reserve(self, scope_key: str, tokens: int, cost_usd: float = 0.0) -> Reservation | None:
        """Check the call against used plus already-reserved usage, and hold
        its estimate. There is no await between the check and the hold, so
        within one event loop concurrent calls can't both pass the check."""
        limits = self._limits.get(scope_key)
        if not limits:
            return None
        used = self._usage.get(scope_key, ExecutionBudgetUsage())
        held = self._reserved.setdefault(scope_key, ExecutionBudgetUsage())

        calls = used.model_calls + held.model_calls + 1
        if limits.max_model_calls is not None and calls > limits.max_model_calls:
            raise BudgetExceededError(
                message="Max model call count reached",
                budget_type="max_model_calls",
                limit=float(limits.max_model_calls),
                current=float(calls - 1),
            )
        total_tokens = used.total_tokens + held.total_tokens + tokens
        if limits.max_tokens is not None and total_tokens > limits.max_tokens:
            raise BudgetExceededError(
                message="Token budget would be exceeded",
                budget_type="max_tokens",
                limit=float(limits.max_tokens),
                current=float(total_tokens),
            )
        total_cost = used.total_cost_usd + held.total_cost_usd + cost_usd
        if limits.max_cost_usd is not None and (
            total_cost > limits.max_cost_usd
            or used.total_cost_usd + held.total_cost_usd >= limits.max_cost_usd
        ):
            raise BudgetExceededError(
                message="Max execution cost budget exceeded",
                budget_type="max_cost_usd",
                limit=limits.max_cost_usd,
                current=total_cost,
            )

        held.model_calls += 1
        held.total_tokens += tokens
        held.total_cost_usd += cost_usd
        return Reservation(scope_key=scope_key, tokens=tokens, cost_usd=cost_usd)

    def _unhold(self, reservation: Reservation) -> None:
        held = self._reserved.get(reservation.scope_key)
        if held is None:
            return  # the scope ended while the call was in flight
        held.model_calls -= 1
        held.total_tokens -= reservation.tokens
        held.total_cost_usd -= reservation.cost_usd

    def reconcile(self, reservation: Reservation | None, actual: ModelUsage) -> None:
        """Replace the call's estimate with what it actually used."""
        if reservation is None:
            return
        self._unhold(reservation)
        self.record_usage(reservation.scope_key, actual)

    def release(self, reservation: Reservation | None) -> None:
        """Drop the estimate of a call that failed before it used anything."""
        if reservation is not None:
            self._unhold(reservation)

    def check_pre_invocation(self, scope_key: str, requested_tokens: int = 0) -> None:
        limits = self._limits.get(scope_key)
        if not limits:
            return

        current = self._usage.get(scope_key, ExecutionBudgetUsage())

        if limits.max_model_calls is not None and current.model_calls >= limits.max_model_calls:
            raise BudgetExceededError(
                message="Max model call count reached",
                budget_type="max_model_calls",
                limit=float(limits.max_model_calls),
                current=float(current.model_calls),
            )

        if limits.max_cost_usd is not None and current.total_cost_usd >= limits.max_cost_usd:
            raise BudgetExceededError(
                message="Max execution cost budget exceeded",
                budget_type="max_cost_usd",
                limit=limits.max_cost_usd,
                current=current.total_cost_usd,
            )

        if (
            limits.max_tokens is not None
            and (current.total_tokens + requested_tokens) > limits.max_tokens
        ):
            raise BudgetExceededError(
                message="Token budget would be exceeded",
                budget_type="max_tokens",
                limit=float(limits.max_tokens),
                current=float(current.total_tokens + requested_tokens),
            )

    def record_usage(self, scope_key: str, usage: ModelUsage) -> None:
        if scope_key not in self._limits:
            return  # unlimited scopes aren't tracked, so nothing accumulates
        current = self._usage.get(scope_key)
        if not current:
            current = ExecutionBudgetUsage()
            self._usage[scope_key] = current

        current.total_tokens += usage.total_tokens
        current.total_cost_usd += usage.estimated_cost_usd
        current.model_calls += 1

    def get_usage(self, scope_key: str) -> ExecutionBudgetUsage:
        return self._usage.get(scope_key, ExecutionBudgetUsage())

    def get_reserved(self, scope_key: str) -> ExecutionBudgetUsage:
        return self._reserved.get(scope_key, ExecutionBudgetUsage())

    def reset(self, scope_key: str | None = None) -> None:
        if scope_key:
            self.end_scope(scope_key)
        else:
            self._limits.clear()
            self._usage.clear()
            self._reserved.clear()
