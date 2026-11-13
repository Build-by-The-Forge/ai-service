"""#57 follow-up: a failed agent run's result carries no exception text.

AgentResult.error_message, and the task result AgentExecutionTaskHandler
returns from it, are fixed text; the error code names the failure's category.
Each test makes an arbitrary exception carrying a sentinel happen where the
text used to flow through, and checks the sentinel can't be recovered from
the result.
"""

import json
from dataclasses import asdict

import pytest

from app.core.agents.contracts import AgentContext, AgentState
from app.core.execution_context import AIExecutionContext
from app.core.models.gateway import ModelGateway
from app.core.models.providers.mock import MockProviderAdapter
from app.services.agents.executor import (
    MODEL_FAILURES_MESSAGE,
    RESOLUTION_FAILED_MESSAGE,
    AgentExecutor,
)
from app.services.tasks.handlers import AgentExecutionTaskHandler

SENTINEL = "student-passage-9a3b"


def agent_context() -> AgentContext:
    return AgentContext(
        tenant_id="school_futa",
        execution_id="exec_agent",
        task_id="task_agent",
        correlation_id="corr_agent",
        permissions=["curriculum:read", "content:read", "assessment:generate"],
        capabilities=["curriculum_intelligence", "content_search", "cbt_generation"],
    )


def executor() -> AgentExecutor:
    return AgentExecutor(
        model_gateway=ModelGateway(
            providers=[MockProviderAdapter(name="openai")], approved_providers={"openai"}
        )
    )


def recoverable(result) -> str:
    return json.dumps(asdict(result), default=str)


@pytest.mark.asyncio
async def test_a_resolution_failure_reports_its_category_not_its_text(monkeypatch):
    agent = executor()

    def broken(agent_id, context):
        raise RuntimeError(f"registry said: {SENTINEL}")

    monkeypatch.setattr(agent, "_resolve_agent", broken)

    result = await agent.run(agent_id="study.assistant", prompt="q", context=agent_context())

    assert result.state == AgentState.FAILED
    assert result.error_code == "AUTHORIZATION_FAILED"
    assert result.error_message == RESOLUTION_FAILED_MESSAGE
    assert SENTINEL not in recoverable(result)


@pytest.mark.asyncio
async def test_repeated_model_failures_report_a_fixed_message(monkeypatch):
    agent = executor()

    async def broken(request):
        raise RuntimeError(f"while building the request: {SENTINEL}")

    monkeypatch.setattr(agent.model_gateway, "generate", broken)

    result = await agent.run(agent_id="study.assistant", prompt="q", context=agent_context())

    assert result.state == AgentState.FAILED
    assert result.error_message == MODEL_FAILURES_MESSAGE
    assert SENTINEL not in recoverable(result)


@pytest.mark.asyncio
async def test_the_task_result_holds_no_exception_text(monkeypatch):
    """What AgentExecutionTaskHandler returns for the task is built from the
    result: the sentinel can't be recovered from it either."""
    agent = executor()

    def broken(agent_id, context):
        raise RuntimeError(f"registry said: {SENTINEL}")

    monkeypatch.setattr(agent, "_resolve_agent", broken)
    ctx = AIExecutionContext(
        execution_id="exec_task",
        tenant_id="6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c",
        job_id="job_1",
        asset_id="asset_1",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_1",
        task_type="agent.execution",
    )

    stored = await AgentExecutionTaskHandler(agent_executor=agent).handle(
        ctx, {"agent_id": "study.assistant", "prompt": "q"}
    )

    assert stored["status"] == AgentState.FAILED.value
    assert stored["error_message"] == RESOLUTION_FAILED_MESSAGE
    assert SENTINEL not in json.dumps(stored, default=str)
