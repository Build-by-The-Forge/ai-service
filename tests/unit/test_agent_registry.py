from app.core.agents.contracts import AgentContext, AgentDefinition
from app.core.agents.registry import (
    AgentRegistry,
)


def test_agent_registry_builtins_and_registration():
    registry = AgentRegistry(register_builtins=True)

    study_agent = registry.get("study.assistant")
    assert study_agent is not None
    assert study_agent.agent_id == "study.assistant"
    assert "curriculum.lookup" in study_agent.allowed_tools

    content_agent = registry.get("content.analyzer")
    assert content_agent is not None
    assert content_agent.agent_id == "content.analyzer"

    custom_agent = AgentDefinition(
        agent_id="custom.agent",
        name="Custom Agent",
        version="1.0.0",
        description="Custom test agent",
        system_prompt="Custom prompt",
    )
    registry.register(custom_agent)
    assert registry.get("custom.agent") is not None

    registry.unregister("custom.agent")
    assert registry.get("custom.agent") is None


def test_agent_registry_permission_and_capability_filtering():
    registry = AgentRegistry(register_builtins=True)

    # Actor has permissions for study.assistant and content.analyzer
    context_authorized = AgentContext(
        tenant_id="school_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
        permissions=["curriculum:read", "content:read", "assessment:generate"],
        capabilities=["curriculum_intelligence", "content_search", "cbt_generation"],
    )
    available = registry.list_available(context_authorized)
    available_ids = [a.agent_id for a in available]
    assert "study.assistant" in available_ids
    assert "content.analyzer" in available_ids

    # Actor has only content:read permission
    context_restricted = AgentContext(
        tenant_id="school_1",
        execution_id="exec_2",
        task_id="task_2",
        correlation_id="corr_2",
        permissions=["content:read"],
        capabilities=["content_search"],
    )
    available_restricted = registry.list_available(context_restricted)
    restricted_ids = [a.agent_id for a in available_restricted]
    assert "content.analyzer" in restricted_ids
    assert "study.assistant" not in restricted_ids  # missing curriculum:read, assessment:generate
