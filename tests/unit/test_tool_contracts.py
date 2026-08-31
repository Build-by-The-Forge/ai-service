import time

from app.core.tools.contracts import (
    ToolContext,
    ToolDefinition,
    ToolExecutionPolicy,
    ToolSideEffect,
)


def test_tool_context_expiry_and_cancellation():
    ctx = ToolContext(
        tenant_id="school_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
        deadline=time.time() + 60,
    )
    assert ctx.is_expired() is False
    assert ctx.is_cancelled() is False

    # Expired context
    expired_ctx = ToolContext(
        tenant_id="school_1",
        execution_id="exec_exp",
        task_id="task_exp",
        correlation_id="corr_exp",
        deadline=time.time() - 10,
    )
    assert expired_ctx.is_expired() is True


def test_tool_definition_to_openai_tool():
    defn = ToolDefinition(
        name="test.lookup",
        version="1.0.0",
        description="Lookup test data",
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
        side_effect=ToolSideEffect.READ_ONLY,
    )

    openai_schema = defn.to_openai_tool()
    assert openai_schema["type"] == "function"
    assert openai_schema["function"]["name"] == "test.lookup"
    assert openai_schema["function"]["description"] == "Lookup test data"
    assert openai_schema["function"]["parameters"]["required"] == ["id"]


def test_tool_execution_policy_defaults():
    policy = ToolExecutionPolicy()
    assert policy.max_tool_calls == 10
    assert policy.max_runtime_seconds == 60.0
    assert policy.allow_proposals is True
    assert policy.allow_mutating is False
