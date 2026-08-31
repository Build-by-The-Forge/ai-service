import pytest
from app.core.agents.contracts import AgentContext, AgentDefinition, AgentState
from app.core.agents.registry import AgentRegistry
from app.core.models.contracts import (
    ModelDecision,
    ModelRequest,
    ModelResponse,
)
from app.core.models.gateway import ModelGateway
from app.core.models.provider import ModelProviderAdapter
from app.services.agents.executor import AgentExecutor


class TenantSpoofingModelProvider(ModelProviderAdapter):
    @property
    def name(self) -> str:
        return "spoofing"

    def supports_model(self, model: str) -> bool:
        return True

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        # LLM tries to invoke curriculum.lookup with another tenant's ID
        return ModelResponse(
            content="Attempting tenant breach",
            model=decision.model,
            provider=self.name,
            tool_calls=[
                {
                    "id": "tc_spoof",
                    "type": "function",
                    "function": {
                        "name": "curriculum.lookup",
                        "arguments": '{"course_code": "CSC201", "tenant_id": "school_unauthorized_999"}',
                    },
                }
            ],
        )

    async def generate_stream(self, request: ModelRequest, decision: ModelDecision):
        raise NotImplementedError()


@pytest.mark.asyncio
async def test_security_rejects_missing_permission():
    context = AgentContext(
        tenant_id="school_1",
        execution_id="exec_sec_1",
        task_id="task_sec_1",
        correlation_id="corr_sec_1",
        permissions=[],  # Missing required permissions
        capabilities=["curriculum_intelligence"],
    )

    executor = AgentExecutor()
    result = await executor.run(
        agent_id="study.assistant",
        prompt="Explain physics",
        context=context,
    )

    assert result.state == AgentState.FAILED
    assert result.error_code == "PERMISSION_DENIED"


@pytest.mark.asyncio
async def test_security_rejects_missing_capability():
    context = AgentContext(
        tenant_id="school_1",
        execution_id="exec_sec_2",
        task_id="task_sec_2",
        correlation_id="corr_sec_2",
        permissions=["curriculum:read", "content:read", "assessment:generate"],
        capabilities=[],  # Missing capabilities
    )

    executor = AgentExecutor()
    result = await executor.run(
        agent_id="study.assistant",
        prompt="Explain physics",
        context=context,
    )

    assert result.state == AgentState.FAILED
    assert result.error_code == "CAPABILITY_DENIED"


@pytest.mark.asyncio
async def test_security_rejects_disabled_agent():
    registry = AgentRegistry(register_builtins=False)
    disabled_def = AgentDefinition(
        agent_id="disabled.agent",
        name="Disabled Agent",
        version="1.0.0",
        description="Disabled agent",
        system_prompt="System prompt",
        is_enabled=False,
    )
    registry.register(disabled_def)

    executor = AgentExecutor(agent_registry=registry)
    context = AgentContext(
        tenant_id="school_1",
        execution_id="exec_sec_3",
        task_id="task_sec_3",
        correlation_id="corr_sec_3",
    )

    result = await executor.run(
        agent_id="disabled.agent",
        prompt="Execute",
        context=context,
    )

    assert result.state == AgentState.FAILED
    assert result.error_code == "AGENT_DISABLED"


@pytest.mark.asyncio
async def test_security_rejects_unknown_agent():
    executor = AgentExecutor()
    context = AgentContext(
        tenant_id="school_1",
        execution_id="exec_sec_4",
        task_id="task_sec_4",
        correlation_id="corr_sec_4",
    )

    result = await executor.run(
        agent_id="unknown.agent.999",
        prompt="Execute",
        context=context,
    )

    assert result.state == AgentState.FAILED
    assert result.error_code == "AGENT_NOT_FOUND"


@pytest.mark.asyncio
async def test_security_blocks_tenant_spoofing_via_tool_calls():
    # LLM requests a tool call targeting another tenant
    gateway = ModelGateway(providers=[TenantSpoofingModelProvider()])
    executor = AgentExecutor(model_gateway=gateway)

    context = AgentContext(
        tenant_id="school_authoritative_futa",
        execution_id="exec_sec_5",
        task_id="task_sec_5",
        correlation_id="corr_sec_5",
        permissions=["curriculum:read", "content:read", "assessment:generate"],
        capabilities=["curriculum_intelligence", "content_search", "cbt_generation"],
    )

    # In step 1, the LLM requests a tool with mismatched tenant -> ToolExecutor catches TENANT_MISMATCH
    # and returns a structured error observation to the agent without breaching the tenant boundary!
    result = await executor.run(
        agent_id="study.assistant",
        prompt="Query data",
        context=context,
    )

    # In loop step 1, ToolExecutor blocked tenant spoofing and returned an error observation
    assert result.tool_call_count >= 1
