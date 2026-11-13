"""#53: a tool failure reaches the agent as a structured failure, never as text.

An exception from a tool can carry student text, a retrieved passage, a
provider's response or an internal detail. The agent gets a stable code, a
retry flag from the failure's category, and a fixed message; each test plants a
sentinel in the exception and checks it never crosses into the serialized
result, the agent's context or the logs.
"""

import asyncio
import json
import logging
from dataclasses import asdict

import pytest

from app.core.agents.contracts import AgentContext
from app.core.models.contracts import ModelDecision, ModelRequest, ModelResponse, ModelUsage
from app.core.models.gateway import ModelGateway
from app.core.models.provider import ModelProviderAdapter
from app.core.tools.contracts import ToolContext, ToolDefinition, ToolInvocation
from app.core.tools.errors import ToolError, ToolTenantMismatchError
from app.core.tools.registry import ToolRegistry
from app.services.agents.executor import AgentExecutor
from app.services.tools.builtins.curriculum import CURRICULUM_LOOKUP_DEFINITION
from app.services.tools.executor import FAILURE_MESSAGES, ToolExecutor

SENTINEL = "student-passage-7e91"
CONTEXT = ToolContext(
    tenant_id="school_futa",
    execution_id="exec_1",
    task_id="task_1",
    correlation_id="corr_1",
    permissions=["curriculum:read", "content:read", "assessment:generate"],
    capabilities=["curriculum_intelligence", "content_search", "cbt_generation"],
    denied_assets=frozenset(),
)


def executor_with(handler, *, timeout_seconds: float = 5.0) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="probe.tool",
            version="1.0.0",
            description="probe",
            input_schema={"type": "object"},
            timeout_seconds=timeout_seconds,
        ),
        handler,
    )
    return ToolExecutor(registry=registry, register_builtins=False)


async def run(executor: ToolExecutor, arguments: dict | None = None):
    return await executor.execute(
        ToolInvocation(tool_name="probe.tool", arguments=arguments or {}, context=CONTEXT)
    )


def serialized(result) -> str:
    return json.dumps(asdict(result)) + json.dumps(result.failure())


@pytest.mark.asyncio
async def test_an_unexpected_exception_is_a_structured_failure_without_its_text(caplog):
    async def crashes(ctx, args):
        raise RuntimeError(f"could not read: {SENTINEL}")

    with caplog.at_level(logging.DEBUG):
        result = await run(executor_with(crashes))

    assert result.failure() == {
        "ok": False,
        "error": {
            "code": "TOOL_EXECUTION_FAILED",
            "retryable": False,
            "message": "The tool could not complete this operation.",
        },
    }
    assert SENTINEL not in serialized(result)
    assert all(SENTINEL not in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_typed_failure_keeps_its_category_code_but_not_its_text():
    async def refuses(ctx, args):
        raise ToolTenantMismatchError(
            "probe.tool", attempted_tenant=SENTINEL, authoritative_tenant="school_futa"
        )

    result = await run(executor_with(refuses))

    assert (result.error_code, result.error_message) == (
        "TENANT_MISMATCH",
        FAILURE_MESSAGES["TENANT_MISMATCH"],
    )
    assert result.retryable is False
    assert SENTINEL not in serialized(result)


@pytest.mark.asyncio
async def test_a_typed_failure_with_an_unknown_code_gets_the_generic_message():
    async def fails(ctx, args):
        raise ToolError(f"internal: {SENTINEL}", error_code="SOMETHING_NEW")

    result = await run(executor_with(fails))

    assert result.error_code == "SOMETHING_NEW"
    assert result.error_message == FAILURE_MESSAGES["TOOL_EXECUTION_FAILED"]
    assert SENTINEL not in serialized(result)


@pytest.mark.asyncio
async def test_an_argument_refused_at_the_tenant_boundary_is_not_quoted_back():
    """The validator's refusal used to quote the attempted tenant, an argument
    the model chose, back into the agent's context."""

    async def never_called(ctx, args):  # pragma: no cover - refused before it runs
        raise AssertionError("ran")

    result = await run(executor_with(never_called), {"tenant_id": SENTINEL})

    assert result.error_code == "TENANT_MISMATCH"
    assert result.error_message == FAILURE_MESSAGES["TENANT_MISMATCH"]
    assert SENTINEL not in serialized(result)


@pytest.mark.asyncio
async def test_a_timeout_is_retryable_by_its_category():
    async def stalls(ctx, args):
        await asyncio.sleep(10)

    result = await run(executor_with(stalls, timeout_seconds=0.05))

    assert result.failure()["error"] == {
        "code": "TOOL_TIMEOUT",
        "retryable": True,
        "message": FAILURE_MESSAGES["TOOL_TIMEOUT"],
    }


class Scripted(ModelProviderAdapter):
    """Asks for one tool call, then answers; records what it was sent."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    @property
    def name(self) -> str:
        return "openai"  # the built-in agents prefer it

    def supports_model(self, model: str) -> bool:
        return True

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        self.requests.append(request)
        calls = (
            [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {
                        "name": "curriculum.lookup",
                        "arguments": '{"course_code": "CSC201"}',
                    },
                }
            ]
            if len(self.requests) == 1
            else None
        )
        return ModelResponse(
            content="done",
            model=decision.model,
            provider="openai",
            tool_calls=calls,
            usage=ModelUsage(),
        )

    async def generate_stream(self, request, decision):  # pragma: no cover - not used
        raise NotImplementedError
        yield


@pytest.mark.asyncio
async def test_the_agent_reads_the_structured_failure_and_nothing_the_exception_said():
    async def crashes(ctx, args):
        raise RuntimeError(f"provider body: {SENTINEL}")

    registry = ToolRegistry()
    registry.register(CURRICULUM_LOOKUP_DEFINITION, crashes)
    model = Scripted()
    agent = AgentExecutor(
        model_gateway=ModelGateway(providers=[model], approved_providers={"openai"}),
        tool_executor=ToolExecutor(registry=registry, register_builtins=False),
    )
    context = AgentContext(
        tenant_id="school_futa",
        execution_id="exec_agent",
        task_id="task_agent",
        correlation_id="corr_agent",
        permissions=list(CONTEXT.permissions),
        capabilities=list(CONTEXT.capabilities),
    )

    await agent.run(agent_id="study.assistant", prompt="What is in CSC201?", context=context)

    # The model's second turn holds the tool's observation.
    second = model.requests[1].messages
    [observation] = [m for m in second if m.name == "curriculum.lookup"]
    assert json.loads(observation.content) == {
        "ok": False,
        "error": {
            "code": "TOOL_EXECUTION_FAILED",
            "retryable": False,
            "message": FAILURE_MESSAGES["TOOL_EXECUTION_FAILED"],
        },
    }
    assert all(SENTINEL not in (m.content or "") for m in second)
