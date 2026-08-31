import pytest
from app.core.agents.contracts import AgentContext, AgentState
from app.core.models.contracts import ModelDecision, ModelRequest, ModelResponse
from app.core.models.provider import ModelProviderAdapter
from app.services.agents.executor import AgentExecutor


class CancellableMockProvider(ModelProviderAdapter):
    @property
    def name(self) -> str:
        return "cancellable"

    def supports_model(self, model: str) -> bool:
        return True

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        return ModelResponse(
            content="Tool call to cancel",
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
async def test_agent_pre_start_cancellation():
    context = AgentContext(
        tenant_id="school_1",
        execution_id="exec_cancel_1",
        task_id="task_cancel_1",
        correlation_id="corr_cancel_1",
        permissions=["curriculum:read", "content:read", "assessment:generate"],
        capabilities=["curriculum_intelligence", "content_search", "cbt_generation"],
        _is_cancelled=True,
        _cancellation_reason="User aborted request",
    )

    executor = AgentExecutor()
    result = await executor.run(
        agent_id="study.assistant",
        prompt="Explain physics",
        context=context,
    )

    assert result.state == AgentState.CANCELLED
    assert result.error_message == "User aborted request"
    assert result.step_count == 0
