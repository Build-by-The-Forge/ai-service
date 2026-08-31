from dataclasses import dataclass
from typing import Dict, Optional

from app.core.tools.contracts import ToolExecutionPolicy
from app.core.tools.errors import ToolBudgetExceededError


@dataclass
class ToolExecutionUsage:
    tool_calls: int = 0
    cumulative_runtime_seconds: float = 0.0


class ToolBudgetTracker:
    """
    Tracks and enforces tool call counts and cumulative execution runtimes per execution context.
    """

    def __init__(self):
        self._usage: Dict[str, ToolExecutionUsage] = {}

    def check_pre_invocation(
        self, scope_key: str, policy: Optional[ToolExecutionPolicy] = None
    ) -> None:
        if not policy:
            return

        current = self._usage.get(scope_key, ToolExecutionUsage())

        if policy.max_tool_calls is not None and current.tool_calls >= policy.max_tool_calls:
            raise ToolBudgetExceededError(
                message=f"Tool invocation budget exceeded (limit={policy.max_tool_calls}, current={current.tool_calls})",
                budget_type="max_tool_calls",
                limit=float(policy.max_tool_calls),
                current=float(current.tool_calls),
            )

        if (
            policy.max_runtime_seconds is not None
            and current.cumulative_runtime_seconds >= policy.max_runtime_seconds
        ):
            raise ToolBudgetExceededError(
                message=f"Tool cumulative runtime budget exceeded (limit={policy.max_runtime_seconds}s, current={current.cumulative_runtime_seconds:.2f}s)",
                budget_type="max_runtime_seconds",
                limit=policy.max_runtime_seconds,
                current=current.cumulative_runtime_seconds,
            )

    def record_invocation(self, scope_key: str, latency_seconds: float) -> None:
        current = self._usage.get(scope_key)
        if not current:
            current = ToolExecutionUsage()
            self._usage[scope_key] = current

        current.tool_calls += 1
        current.cumulative_runtime_seconds += latency_seconds

    def get_usage(self, scope_key: str) -> ToolExecutionUsage:
        return self._usage.get(scope_key, ToolExecutionUsage())

    def reset(self, scope_key: Optional[str] = None) -> None:
        if scope_key:
            self._usage.pop(scope_key, None)
        else:
            self._usage.clear()
