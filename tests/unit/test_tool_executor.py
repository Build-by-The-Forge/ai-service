import asyncio

import pytest

from app.core.tools.budget import ToolBudgetTracker
from app.core.tools.contracts import (
    ToolContext,
    ToolDefinition,
    ToolExecutionPolicy,
    ToolInvocation,
)
from app.core.tools.registry import ToolRegistry
from app.services.tools.executor import ToolExecutor


@pytest.fixture
def auth_context():
    return ToolContext(
        tenant_id="school_futa",
        execution_id="exec_tool_1",
        task_id="task_tool_1",
        correlation_id="corr_tool_1",
        permissions=["curriculum:read", "content:read", "assessment:generate"],
        capabilities=["curriculum_intelligence", "content_search", "cbt_generation"],
    )


@pytest.mark.asyncio
async def test_executor_successful_curriculum_lookup(auth_context):
    executor = ToolExecutor()

    invocation = ToolInvocation(
        tool_name="curriculum.lookup",
        arguments={"course_code": "csc201"},
        context=auth_context,
    )

    result = await executor.execute(invocation)
    assert result.success is True
    assert result.tool_name == "curriculum.lookup"
    assert result.data["course_code"] == "CSC201"
    assert len(result.data["units"]) >= 1
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_executor_successful_content_search(auth_context):
    executor = ToolExecutor()

    invocation = ToolInvocation(
        tool_name="content.search",
        arguments={"query": "algorithms and complexity", "limit": 2},
        context=auth_context,
    )

    result = await executor.execute(invocation)
    assert result.success is True
    assert result.data["query"] == "algorithms and complexity"
    assert len(result.data["results"]) <= 2


@pytest.mark.asyncio
async def test_executor_successful_draft_question_proposal(auth_context):
    executor = ToolExecutor()

    invocation = ToolInvocation(
        tool_name="draft.question",
        arguments={"topic": "Binary Search Trees", "difficulty": "MEDIUM"},
        context=auth_context,
    )

    result = await executor.execute(invocation)
    assert result.success is True
    assert result.data["status"] == "PROPOSAL_CREATED"
    assert result.data["proposal_id"].startswith("prop_")
    assert result.data["topic"] == "Binary Search Trees"
    assert "AI Proposal" in result.data["disclaimer"]


@pytest.mark.asyncio
async def test_executor_handles_tool_timeout(auth_context):
    registry = ToolRegistry()

    async def slow_handler(ctx, args):
        await asyncio.sleep(0.5)
        return {"data": "done"}

    slow_defn = ToolDefinition(
        name="slow.tool",
        version="1.0.0",
        description="Slow tool",
        input_schema={"type": "object"},
        timeout_seconds=0.05,
    )
    registry.register(slow_defn, slow_handler)

    executor = ToolExecutor(registry=registry, register_builtins=False)

    invocation = ToolInvocation(
        tool_name="slow.tool",
        arguments={},
        context=auth_context,
    )

    result = await executor.execute(invocation)
    assert result.success is False
    assert result.error_code == "TOOL_TIMEOUT"


@pytest.mark.asyncio
async def test_executor_failure_isolation_on_runtime_crash(auth_context):
    registry = ToolRegistry()

    async def crashing_handler(ctx, args):
        raise ValueError("Simulated unexpected database crash")

    crash_defn = ToolDefinition(
        name="crash.tool",
        version="1.0.0",
        description="Crashing tool",
        input_schema={"type": "object"},
    )
    registry.register(crash_defn, crashing_handler)

    executor = ToolExecutor(registry=registry, register_builtins=False)

    invocation = ToolInvocation(
        tool_name="crash.tool",
        arguments={},
        context=auth_context,
    )

    result = await executor.execute(invocation)
    assert result.success is False
    assert result.error_code == "EXECUTION_FAILED"
    assert "Simulated unexpected database crash" in result.error_message


@pytest.mark.asyncio
async def test_executor_enforces_budget_limits(auth_context):
    registry = ToolRegistry()

    async def quick_handler(ctx, args):
        return {"status": "ok"}

    quick_defn = ToolDefinition(
        name="quick.tool",
        version="1.0.0",
        description="Quick tool",
        input_schema={"type": "object"},
    )
    registry.register(quick_defn, quick_handler)

    tracker = ToolBudgetTracker()
    executor = ToolExecutor(registry=registry, budget_tracker=tracker, register_builtins=False)

    policy = ToolExecutionPolicy(max_tool_calls=1)

    invocation = ToolInvocation(
        tool_name="quick.tool",
        arguments={},
        context=auth_context,
    )

    # 1st execution succeeds
    res1 = await executor.execute(invocation, policy=policy)
    assert res1.success is True

    # 2nd execution rejected by budget
    res2 = await executor.execute(invocation, policy=policy)
    assert res2.success is False
    assert res2.error_code == "TOOL_BUDGET_EXCEEDED"
