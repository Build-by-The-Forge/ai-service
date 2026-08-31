import pytest
from app.core.agents.contracts import (
    AgentContext,
    AgentExecutionPolicy,
    AgentState,
)
from app.core.models.contracts import (
    ModelDecision,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)
from app.core.models.gateway import ModelGateway
from app.core.models.provider import ModelProviderAdapter
from app.services.agents.executor import AgentExecutor


class LoopingToolModelProvider(ModelProviderAdapter):
    """Provider that continuously issues tool calls to test budget exhaustion."""

    @property
    def name(self) -> str:
        return "looping"

    def supports_model(self, model: str) -> bool:
        return True

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        return ModelResponse(
            content="Looking up again...",
            model=decision.model,
            provider=self.name,
            tool_calls=[
                {
                    "id": "tc_loop",
                    "type": "function",
                    "function": {
                        "name": "curriculum.lookup",
                        "arguments": '{"course_code": "CSC201"}',
                    },
                }
            ],
            usage=ModelUsage(
                prompt_tokens=50, completion_tokens=20, total_tokens=70, estimated_cost_usd=0.01
            ),
        )

    async def generate_stream(self, request: ModelRequest, decision: ModelDecision):
        raise NotImplementedError()


@pytest.fixture
def auth_context():
    return AgentContext(
        tenant_id="school_1",
        execution_id="exec_budget_1",
        task_id="task_budget_1",
        correlation_id="corr_budget_1",
        permissions=["curriculum:read", "content:read", "assessment:generate"],
        capabilities=["curriculum_intelligence", "content_search", "cbt_generation"],
    )


@pytest.mark.asyncio
async def test_agent_max_steps_budget_exceeded(auth_context):
    gateway = ModelGateway(providers=[LoopingToolModelProvider()])
    executor = AgentExecutor(model_gateway=gateway)

    policy = AgentExecutionPolicy(max_steps=2, max_model_calls=10, max_tool_calls=10)

    result = await executor.run(
        agent_id="study.assistant",
        prompt="Infinite reasoning",
        context=auth_context,
        custom_policy=policy,
    )

    assert result.state == AgentState.MAX_STEPS_EXCEEDED
    assert result.step_count == 2


@pytest.mark.asyncio
async def test_agent_max_tool_calls_budget_exceeded(auth_context):
    gateway = ModelGateway(providers=[LoopingToolModelProvider()])
    executor = AgentExecutor(model_gateway=gateway)

    # Tool call budget of 1: will execute 1 tool call then trip budget on step 2
    policy = AgentExecutionPolicy(max_steps=10, max_model_calls=10, max_tool_calls=1)

    result = await executor.run(
        agent_id="study.assistant",
        prompt="Too many tools",
        context=auth_context,
        custom_policy=policy,
    )

    assert result.state == AgentState.BUDGET_EXCEEDED
    assert result.tool_call_count == 1


@pytest.mark.asyncio
async def test_agent_max_tokens_budget_exceeded(auth_context):
    gateway = ModelGateway(providers=[LoopingToolModelProvider()])
    executor = AgentExecutor(model_gateway=gateway)

    # Token budget of 100 tokens: each step adds 70 tokens
    policy = AgentExecutionPolicy(max_steps=10, max_tokens=100)

    result = await executor.run(
        agent_id="study.assistant",
        prompt="Too many tokens",
        context=auth_context,
        custom_policy=policy,
    )

    assert result.state == AgentState.BUDGET_EXCEEDED
    assert result.usage.total_tokens >= 70


@pytest.mark.asyncio
async def test_agent_max_cost_budget_exceeded(auth_context):
    gateway = ModelGateway(providers=[LoopingToolModelProvider()])
    executor = AgentExecutor(model_gateway=gateway)

    # Cost budget of $0.015: each step costs $0.01
    policy = AgentExecutionPolicy(max_steps=10, max_cost_usd=0.015)

    result = await executor.run(
        agent_id="study.assistant",
        prompt="Cost ceiling",
        context=auth_context,
        custom_policy=policy,
    )

    assert result.state == AgentState.BUDGET_EXCEEDED
    assert result.estimated_cost_usd >= 0.01
