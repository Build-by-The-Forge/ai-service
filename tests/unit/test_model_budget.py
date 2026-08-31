import pytest

from app.core.models.budget import BudgetLimits, BudgetTracker
from app.core.models.contracts import ModelUsage
from app.core.models.errors import BudgetExceededError


def test_budget_tracker_enforces_max_model_calls():
    tracker = BudgetTracker()
    tracker.set_budget("exec_1", BudgetLimits(max_model_calls=2))

    # 1st call check & record
    tracker.check_pre_invocation("exec_1")
    tracker.record_usage("exec_1", ModelUsage(total_tokens=10))

    # 2nd call check & record
    tracker.check_pre_invocation("exec_1")
    tracker.record_usage("exec_1", ModelUsage(total_tokens=10))

    # 3rd call should fail budget check
    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.check_pre_invocation("exec_1")
    assert exc_info.value.budget_type == "max_model_calls"


def test_budget_tracker_enforces_max_cost():
    tracker = BudgetTracker()
    tracker.set_budget("exec_2", BudgetLimits(max_cost_usd=0.01))

    # Exceed cost limit
    tracker.record_usage("exec_2", ModelUsage(estimated_cost_usd=0.015))

    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.check_pre_invocation("exec_2")
    assert exc_info.value.budget_type == "max_cost_usd"


def test_budget_tracker_enforces_max_tokens():
    tracker = BudgetTracker()
    tracker.set_budget("exec_3", BudgetLimits(max_tokens=100))

    tracker.record_usage("exec_3", ModelUsage(total_tokens=80))

    # Requesting 30 tokens when current is 80 (80+30 > 100)
    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.check_pre_invocation("exec_3", requested_tokens=30)
    assert exc_info.value.budget_type == "max_tokens"
