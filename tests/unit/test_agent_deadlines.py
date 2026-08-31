import time
import pytest
from app.core.agents.contracts import AgentContext, AgentExecutionPolicy, AgentState
from app.core.models.contracts import ModelDecision, ModelRequest, ModelResponse
from app.core.models.gateway import ModelGateway
from app.core.models.provider import ModelProviderAdapter
from app.services.agents.executor import AgentExecutor


class SlowMockProvider(ModelProviderAdapter):
    @property
    def name(self) -> str:
        return "slow"

    def supports_model(self, model: str) -> bool:
        return True

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        import asyncio

        await asyncio.sleep(0.1)
        return ModelResponse(
            content="Slow reasoning step",
            model=decision.model,
            provider=self.name,
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "curriculum.lookup",
                        "arguments": '{"course_code":"CSC101"}',
                    },
                }
            ],
        )

    async def generate_stream(self, request: ModelRequest, decision: ModelDecision):
        raise NotImplementedError()


@pytest.mark.asyncio
async def test_agent_pre_start_expired_deadline():
    context = AgentContext(
        tenant_id="school_1",
        execution_id="exec_dl_1",
        task_id="task_dl_1",
        correlation_id="corr_dl_1",
        permissions=["curriculum:read", "content:read", "assessment:generate"],
        capabilities=["curriculum_intelligence", "content_search", "cbt_generation"],
        deadline=time.time() - 10,  # Expired
    )

    executor = AgentExecutor()
    result = await executor.run(
        agent_id="study.assistant",
        prompt="Explain physics",
        context=context,
    )

    assert result.state == AgentState.TIMED_OUT
    assert result.step_count == 0


@pytest.mark.asyncio
async def test_agent_runtime_limit_exceeded():
    gateway = ModelGateway(providers=[SlowMockProvider()])
    executor = AgentExecutor(model_gateway=gateway)

    context = AgentContext(
        tenant_id="school_1",
        execution_id="exec_dl_2",
        task_id="task_dl_2",
        correlation_id="corr_dl_2",
        permissions=["curriculum:read", "content:read", "assessment:generate"],
        capabilities=["curriculum_intelligence", "content_search", "cbt_generation"],
    )

    # Runtime limit of 0.05 seconds while provider sleeps 0.1s
    policy = AgentExecutionPolicy(max_runtime_seconds=0.05)

    result = await executor.run(
        agent_id="study.assistant",
        prompt="Fast timeout",
        context=context,
        custom_policy=policy,
    )

    assert result.state == AgentState.TIMED_OUT
