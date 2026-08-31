import time
import pytest
from app.core.tools.contracts import (
    ToolContext,
    ToolDefinition,
    ToolExecutionPolicy,
    ToolInvocation,
    ToolSideEffect,
)
from app.core.tools.registry import ToolRegistry
from app.services.tools.executor import ToolExecutor


@pytest.fixture
def test_executor():
    return ToolExecutor()


@pytest.mark.asyncio
async def test_security_rejects_spoofed_tenant_id(test_executor):
    context = ToolContext(
        tenant_id="school_authoritative_futa",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
        permissions=["curriculum:read"],
        capabilities=["curriculum_intelligence"],
    )

    invocation = ToolInvocation(
        tool_name="curriculum.lookup",
        arguments={"course_code": "CSC201", "tenant_id": "school_unauthorized_unilag"},
        context=context,
    )

    result = await test_executor.execute(invocation)
    assert result.success is False
    assert result.error_code == "TENANT_MISMATCH"
    assert "school_unauthorized_unilag" in result.error_message


@pytest.mark.asyncio
async def test_security_rejects_missing_permission(test_executor):
    context = ToolContext(
        tenant_id="school_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
        permissions=[],  # Missing curriculum:read
        capabilities=["curriculum_intelligence"],
    )

    invocation = ToolInvocation(
        tool_name="curriculum.lookup",
        arguments={"course_code": "CSC201"},
        context=context,
    )

    result = await test_executor.execute(invocation)
    assert result.success is False
    assert result.error_code == "PERMISSION_DENIED"


@pytest.mark.asyncio
async def test_security_rejects_missing_capability(test_executor):
    context = ToolContext(
        tenant_id="school_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
        permissions=["curriculum:read"],
        capabilities=[],  # Missing curriculum_intelligence
    )

    invocation = ToolInvocation(
        tool_name="curriculum.lookup",
        arguments={"course_code": "CSC201"},
        context=context,
    )

    result = await test_executor.execute(invocation)
    assert result.success is False
    assert result.error_code == "CAPABILITY_DENIED"


@pytest.mark.asyncio
async def test_security_rejects_tool_not_in_policy_allowlist(test_executor):
    context = ToolContext(
        tenant_id="school_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
        permissions=["curriculum:read", "content:read"],
        capabilities=["curriculum_intelligence", "content_search"],
    )

    # Allowed tools only contains curriculum.lookup
    policy = ToolExecutionPolicy(allowed_tools=["curriculum.lookup"])

    # Attempt to invoke content.search
    invocation = ToolInvocation(
        tool_name="content.search",
        arguments={"query": "databases"},
        context=context,
    )

    result = await test_executor.execute(invocation, policy=policy)
    assert result.success is False
    assert result.error_code == "TOOL_NOT_ALLOWED"


@pytest.mark.asyncio
async def test_security_rejects_mutating_tool_when_disallowed():
    registry = ToolRegistry()

    async def mutate_handler(ctx, args):
        return {"mutated": True}

    mutate_defn = ToolDefinition(
        name="database.mutate",
        version="1.0.0",
        description="Mutate database directly",
        input_schema={"type": "object"},
        side_effect=ToolSideEffect.MUTATING,
    )
    registry.register(mutate_defn, mutate_handler)

    executor = ToolExecutor(registry=registry, register_builtins=False)

    context = ToolContext(
        tenant_id="school_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
    )

    invocation = ToolInvocation(
        tool_name="database.mutate",
        arguments={},
        context=context,
    )

    # Policy disallows mutating tools by default (allow_mutating=False)
    result = await executor.execute(invocation, policy=ToolExecutionPolicy(allow_mutating=False))
    assert result.success is False
    assert result.error_code == "MUTATION_DISALLOWED"


@pytest.mark.asyncio
async def test_security_rejects_proposal_tool_when_disallowed(test_executor):
    context = ToolContext(
        tenant_id="school_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
        permissions=["assessment:generate"],
        capabilities=["cbt_generation"],
    )

    invocation = ToolInvocation(
        tool_name="draft.question",
        arguments={"topic": "Graphs", "difficulty": "EASY"},
        context=context,
    )

    result = await test_executor.execute(
        invocation, policy=ToolExecutionPolicy(allow_proposals=False)
    )
    assert result.success is False
    assert result.error_code == "PROPOSALS_DISALLOWED"


@pytest.mark.asyncio
async def test_security_rejects_cancelled_context(test_executor):
    context = ToolContext(
        tenant_id="school_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
        permissions=["curriculum:read"],
        capabilities=["curriculum_intelligence"],
        _is_cancelled=True,
    )

    invocation = ToolInvocation(
        tool_name="curriculum.lookup",
        arguments={"course_code": "CSC201"},
        context=context,
    )

    result = await test_executor.execute(invocation)
    assert result.success is False
    assert result.error_code == "EXECUTION_CANCELLED"


@pytest.mark.asyncio
async def test_security_rejects_expired_deadline(test_executor):
    context = ToolContext(
        tenant_id="school_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
        permissions=["curriculum:read"],
        capabilities=["curriculum_intelligence"],
        deadline=time.time() - 10,
    )

    invocation = ToolInvocation(
        tool_name="curriculum.lookup",
        arguments={"course_code": "CSC201"},
        context=context,
    )

    result = await test_executor.execute(invocation)
    assert result.success is False
    assert result.error_code == "DEADLINE_EXPIRED"


@pytest.mark.asyncio
async def test_security_rejects_unknown_tool(test_executor):
    context = ToolContext(
        tenant_id="school_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
    )

    invocation = ToolInvocation(
        tool_name="non_existent.tool",
        arguments={},
        context=context,
    )

    result = await test_executor.execute(invocation)
    assert result.success is False
    assert result.error_code == "TOOL_NOT_FOUND"
