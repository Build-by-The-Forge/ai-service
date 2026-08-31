import pytest

from app.core.agents.contracts import AgentContext, AgentState
from app.core.models.contracts import (
    ModelDecision,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)
from app.core.models.gateway import ModelGateway
from app.core.models.provider import ModelProviderAdapter
from app.services.agents.executor import AgentExecutor


class ScriptedModelProvider(ModelProviderAdapter):
    """
    Mock model adapter that yields pre-programmed ModelResponses in order.
    """

    def __init__(self, responses: list[ModelResponse]):
        self._responses = list(responses)
        self.invocations: list[ModelRequest] = []

    @property
    def name(self) -> str:
        return "scripted"

    def supports_model(self, model: str) -> bool:
        return True

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        self.invocations.append(request)
        if self._responses:
            return self._responses.pop(0)
        return ModelResponse(
            content="Default fallback final", model=decision.model, provider=self.name
        )

    async def generate_stream(self, request: ModelRequest, decision: ModelDecision):
        raise NotImplementedError()


@pytest.fixture
def auth_context():
    return AgentContext(
        tenant_id="school_futa",
        execution_id="exec_agent_1",
        task_id="task_agent_1",
        correlation_id="corr_agent_1",
        permissions=["curriculum:read", "content:read", "assessment:generate"],
        capabilities=["curriculum_intelligence", "content_search", "cbt_generation"],
    )


@pytest.mark.asyncio
async def test_agent_single_step_final_answer(auth_context):
    mock_resp = ModelResponse(
        content="The semester starts on Monday.",
        model="gpt-4o",
        provider="scripted",
        usage=ModelUsage(
            prompt_tokens=15, completion_tokens=8, total_tokens=23, estimated_cost_usd=0.0001
        ),
    )
    provider = ScriptedModelProvider([mock_resp])
    gateway = ModelGateway(providers=[provider])
    executor = AgentExecutor(model_gateway=gateway)

    result = await executor.run(
        agent_id="study.assistant",
        prompt="When does the semester start?",
        context=auth_context,
    )

    assert result.state == AgentState.COMPLETED
    assert result.output == "The semester starts on Monday."
    assert result.step_count == 1
    assert result.model_call_count == 1
    assert result.tool_call_count == 0
    assert result.usage.total_tokens == 23


@pytest.mark.asyncio
async def test_agent_multi_step_tool_orchestration(auth_context):
    # Step 1: Model requests curriculum lookup tool
    step1_resp = ModelResponse(
        content="Let me look up the syllabus for CSC201.",
        model="gpt-4o",
        provider="scripted",
        tool_calls=[
            {
                "id": "tc_1",
                "type": "function",
                "function": {
                    "name": "curriculum.lookup",
                    "arguments": '{"course_code": "CSC201"}',
                },
            }
        ],
        usage=ModelUsage(
            prompt_tokens=20, completion_tokens=10, total_tokens=30, estimated_cost_usd=0.0002
        ),
    )

    # Step 2: Model uses tool observation to produce final answer
    step2_resp = ModelResponse(
        content="CSC201 covers Fundamental Concepts and Advanced Applications.",
        model="gpt-4o",
        provider="scripted",
        tool_calls=None,
        usage=ModelUsage(
            prompt_tokens=40, completion_tokens=15, total_tokens=55, estimated_cost_usd=0.0003
        ),
    )

    provider = ScriptedModelProvider([step1_resp, step2_resp])
    gateway = ModelGateway(providers=[provider])
    executor = AgentExecutor(model_gateway=gateway)

    result = await executor.run(
        agent_id="study.assistant",
        prompt="What topics are in CSC201?",
        context=auth_context,
    )

    assert result.state == AgentState.COMPLETED
    assert "Fundamental Concepts and Advanced Applications" in result.output
    assert result.step_count == 2
    assert result.model_call_count == 2
    assert result.tool_call_count == 1
    assert result.usage.total_tokens == 85
