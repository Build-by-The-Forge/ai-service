import time

import pytest

from app.core.agents.contracts import (
    AgentContext,
    AgentRun,
    AgentState,
)
from app.core.models.contracts import ModelUsage


def test_agent_context_invariants():
    ctx = AgentContext(
        tenant_id="school_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
        deadline=time.time() + 60,
    )
    assert ctx.tenant_id == "school_1"
    assert ctx.is_expired() is False
    assert ctx.is_cancelled() is False


def test_agent_run_monotonic_metrics_and_transitions():
    run = AgentRun(
        run_id="run_1",
        agent_id="test.agent",
        agent_version="1.0.0",
        tenant_id="school_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
    )
    assert run.state == AgentState.CREATED
    assert run.is_active() is True

    # Record model call
    run.record_model_call(
        ModelUsage(
            prompt_tokens=10, completion_tokens=20, total_tokens=30, estimated_cost_usd=0.0005
        )
    )
    assert run.model_call_count == 1
    assert run.total_tokens == 30
    assert run.estimated_cost_usd == 0.0005

    # Record step & tool call
    run.record_step()
    run.record_tool_call()
    assert run.step_count == 1
    assert run.tool_call_count == 1

    # Transition to terminal state
    run.transition_to(AgentState.COMPLETED)
    assert run.state == AgentState.COMPLETED
    assert run.is_active() is False
    assert run.completed_at is not None

    # Cannot transition out of terminal state
    with pytest.raises(ValueError):
        run.transition_to(AgentState.RUNNING)
