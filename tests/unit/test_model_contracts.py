from app.core.models.contracts import (
    ModelInvocationContext,
    ModelMessage,
    ModelPolicy,
    ModelRequest,
    ModelRole,
    ModelUsage,
)
from app.core.models.gateway import ModelGateway


def test_model_message_serialization():
    msg = ModelMessage(
        role=ModelRole.USER,
        content="What is the syllabus for CS101?",
        name="student_1",
    )
    dict_repr = msg.to_dict()
    assert dict_repr["role"] == "user"
    assert dict_repr["content"] == "What is the syllabus for CS101?"
    assert dict_repr["name"] == "student_1"


def test_model_usage_addition():
    u1 = ModelUsage(
        prompt_tokens=10, completion_tokens=20, total_tokens=30, estimated_cost_usd=0.001
    )
    u2 = ModelUsage(
        prompt_tokens=5, completion_tokens=15, total_tokens=20, estimated_cost_usd=0.0005
    )
    u3 = u1 + u2

    assert u3.prompt_tokens == 15
    assert u3.completion_tokens == 35
    assert u3.total_tokens == 50
    assert u3.estimated_cost_usd == 0.0015


def test_model_policy_resolution():
    gateway = ModelGateway()
    context = ModelInvocationContext(
        tenant_id="school_1",
        execution_id="exec_1",
        task_id="task_1",
        correlation_id="corr_1",
    )

    # High quality policy
    req_high = ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="Explain quantum computing")],
        context=context,
        policy=ModelPolicy(quality="high", timeout_seconds=45.0),
    )
    decision_high = gateway.resolve_policy(req_high)
    assert decision_high.model == "gpt-4o"
    assert decision_high.timeout_seconds == 45.0

    # Fast policy
    req_fast = ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="Ping")],
        context=context,
        policy=ModelPolicy(quality="fast"),
    )
    decision_fast = gateway.resolve_policy(req_fast)
    assert decision_fast.model == "gpt-4o-mini"
