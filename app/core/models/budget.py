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


class BudgetTracker:
    """
    Enforces per-tenant and per-execution token, cost, and call-count budgets.
    Prevents uncontrolled reasoning loops and runaway provider charges.
    """

    def __init__(self):
        self._limits: dict[str, BudgetLimits] = {}
        self._usage: dict[str, ExecutionBudgetUsage] = {}

    def set_budget(self, scope_key: str, limits: BudgetLimits) -> None:
        self._limits[scope_key] = limits

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
        current = self._usage.get(scope_key)
        if not current:
            current = ExecutionBudgetUsage()
            self._usage[scope_key] = current

        current.total_tokens += usage.total_tokens
        current.total_cost_usd += usage.estimated_cost_usd
        current.model_calls += 1

    def get_usage(self, scope_key: str) -> ExecutionBudgetUsage:
        return self._usage.get(scope_key, ExecutionBudgetUsage())

    def reset(self, scope_key: str | None = None) -> None:
        if scope_key:
            self._limits.pop(scope_key, None)
            self._usage.pop(scope_key, None)
        else:
            self._limits.clear()
            self._usage.clear()
