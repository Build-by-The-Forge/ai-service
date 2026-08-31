from app.core.tools.contracts import (
    ToolContext,
    ToolDefinition,
    ToolExecutionPolicy,
    ToolSideEffect,
)
from app.core.tools.registry import ToolRegistry


async def dummy_handler(ctx, args):
    return {"status": "ok"}


def test_tool_registry_registration_and_lookup():
    registry = ToolRegistry()
    defn = ToolDefinition(
        name="test.tool",
        version="1.0.0",
        description="A test tool",
        input_schema={"type": "object"},
    )

    registry.register(defn, dummy_handler)
    retrieved = registry.get("test.tool")
    assert retrieved is not None
    assert retrieved[0].name == "test.tool"

    registry.unregister("test.tool")
    assert registry.get("test.tool") is None


def test_tool_registry_policy_aware_filtering():
    registry = ToolRegistry()

    t1 = ToolDefinition(
        name="curriculum.read",
        version="1.0.0",
        description="Read curriculum",
        input_schema={"type": "object"},
        required_permissions=["curriculum:read"],
        required_capabilities=["curriculum_intelligence"],
        side_effect=ToolSideEffect.READ_ONLY,
    )
    t2 = ToolDefinition(
        name="draft.proposal",
        version="1.0.0",
        description="Draft proposal",
        input_schema={"type": "object"},
        required_permissions=["assessment:generate"],
        side_effect=ToolSideEffect.PROPOSAL,
    )
    t3 = ToolDefinition(
        name="course.mutate",
        version="1.0.0",
        description="Mutate course",
        input_schema={"type": "object"},
        required_permissions=["course:write"],
        side_effect=ToolSideEffect.MUTATING,
    )
    t4_disabled = ToolDefinition(
        name="disabled.tool",
        version="1.0.0",
        description="Disabled tool",
        input_schema={"type": "object"},
        is_enabled=False,
    )

    registry.register(t1, dummy_handler)
    registry.register(t2, dummy_handler)
    registry.register(t3, dummy_handler)
    registry.register(t4_disabled, dummy_handler)

    # Actor has curriculum:read and assessment:generate permissions
    context = ToolContext(
        tenant_id="school_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
        permissions=["curriculum:read", "assessment:generate"],
        capabilities=["curriculum_intelligence"],
    )

    # Policy 1: Default policy (allow_proposals=True, allow_mutating=False)
    available = registry.list_available(context, ToolExecutionPolicy())
    available_names = [t.name for t in available]
    assert "curriculum.read" in available_names
    assert "draft.proposal" in available_names
    assert "course.mutate" not in available_names  # Lacks permission AND mutating disallowed
    assert "disabled.tool" not in available_names  # Disabled

    # Policy 2: Explicit allowed_tools allowlist
    policy_restricted = ToolExecutionPolicy(allowed_tools=["curriculum.read"])
    available_restricted = registry.list_available(context, policy_restricted)
    assert len(available_restricted) == 1
    assert available_restricted[0].name == "curriculum.read"

    # Policy 3: Disallow proposals
    policy_no_props = ToolExecutionPolicy(allow_proposals=False)
    available_no_props = registry.list_available(context, policy_no_props)
    assert "draft.proposal" not in [t.name for t in available_no_props]
