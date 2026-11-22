"""ToolContract, AgentTask and ExecutionCheckpoint: validated, versioned,
serialisable, and free of provider and server types."""

import ast
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.agentic.contracts import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATES,
    AgentTask,
    Boundary,
    CallerContext,
    CallKind,
    CallRecord,
    CompletionCriteria,
    EvidenceRequirements,
    ExecutionCheckpoint,
    ExecutionState,
    Idempotency,
    ResultRef,
    RetryClass,
    RetryPolicy,
    SideEffect,
    ToolContract,
    can_transition,
    canonical_json,
    digest,
)

OBJ_IN = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}
OBJ_OUT = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def contract(**over):
    base = dict(
        name="search_course_material",
        version="1.0.0",
        description="Search the student's authorised course material.",
        input_schema=OBJ_IN,
        output_schema=OBJ_OUT,
    )
    base.update(over)
    return ToolContract(**base)


def ctx(**over):
    base = dict(tenant_id="t1", user_id="u1", permissions=("content:read",), denied_assets=())
    base.update(over)
    return CallerContext(**base)


def task(**over):
    base = dict(
        task_id="task1",
        execution_id="exec1",
        task_type="study_plan",
        goal="Plan my week",
        context=ctx(),
        allowed_tools=("search_course_material",),
    )
    base.update(over)
    return AgentTask(**base)


def checkpoint(**over):
    base = dict(
        execution_id="exec1",
        task_id="task1",
        task_digest=task().digest(),
        tenant_id="t1",
        user_id="u1",
    )
    base.update(over)
    return ExecutionCheckpoint(**base)


# --- ToolContract ------------------------------------------------------------


def test_a_valid_read_contract_defaults_are_safe():
    c = contract()
    assert c.side_effect is SideEffect.READ
    assert c.idempotency is Idempotency.NATURAL
    assert c.retry.classification is RetryClass.NEVER
    assert c.boundaries == ()
    assert c.schema_version == 1


@pytest.mark.parametrize("name", ["Search", "9lives", "ab", "has-dash", "x" * 65, "has space"])
def test_bad_names_are_refused(name):
    with pytest.raises(ValidationError):
        contract(name=name)


@pytest.mark.parametrize("version", ["1", "1.0", "v1.0.0", "1.0.0-rc1"])
def test_bad_versions_are_refused(version):
    with pytest.raises(ValidationError):
        contract(version=version)


def test_permissions_and_capabilities_are_checked():
    with pytest.raises(ValidationError):
        contract(required_permissions=("content_read",))
    with pytest.raises(ValidationError):
        contract(required_permissions=("content:read", "content:read"))
    with pytest.raises(ValidationError):
        contract(required_capabilities=("Bad Cap",))


def test_schemas_must_be_enforceable_objects():
    with pytest.raises(ValidationError):
        contract(input_schema={**OBJ_IN, "properties": {"q": {"anyOf": []}}})
    with pytest.raises(ValidationError):
        contract(output_schema={"type": "string"})


def test_input_schema_must_close_additional_properties():
    """A model must not be able to add a tenant or a score to the arguments."""
    loose = {"type": "object", "properties": {"query": {"type": "string"}}}
    with pytest.raises(ValidationError, match="additionalProperties"):
        contract(input_schema=loose)


def test_a_read_is_naturally_idempotent():
    with pytest.raises(ValidationError):
        contract(idempotency=Idempotency.KEYED)


def test_proposals_and_writes_are_keyed():
    with pytest.raises(ValidationError):
        contract(side_effect=SideEffect.PROPOSE)
    ok = contract(side_effect=SideEffect.PROPOSE, idempotency=Idempotency.KEYED)
    assert ok.side_effect is SideEffect.PROPOSE


def test_a_write_goes_through_the_backend_boundary():
    keyed = dict(side_effect=SideEffect.WRITE_THROUGH_BACKEND, idempotency=Idempotency.KEYED)
    with pytest.raises(ValidationError, match="backend"):
        contract(**keyed)
    ok = contract(boundaries=(Boundary.BACKEND,), **keyed)
    assert ok.side_effect is SideEffect.WRITE_THROUGH_BACKEND


def test_a_tool_whose_repeat_is_unsafe_has_no_effect_class_that_allows_it():
    """NONE is reachable by no side-effect class: a read must be NATURAL and a
    propose or write must be KEYED, so a model can never be handed a tool
    the runtime would have to retry unsafely."""
    for effect in SideEffect:
        with pytest.raises(ValidationError):
            contract(
                side_effect=effect,
                idempotency=Idempotency.NONE,
                boundaries=(Boundary.BACKEND,),
            )


def test_retry_policy_is_consistent():
    with pytest.raises(ValidationError):
        RetryPolicy(classification=RetryClass.NEVER, max_retries=1)
    with pytest.raises(ValidationError):
        RetryPolicy(classification=RetryClass.TRANSIENT, max_retries=0)
    with pytest.raises(ValidationError):
        RetryPolicy(classification=RetryClass.TRANSIENT, max_retries=4)
    assert RetryPolicy(classification=RetryClass.TRANSIENT, max_retries=2).max_retries == 2


def test_evidence_comes_from_the_retrieval_boundary():
    with pytest.raises(ValidationError, match="retrieval"):
        contract(produces_evidence=True)
    assert contract(produces_evidence=True, boundaries=(Boundary.RETRIEVAL,)).produces_evidence


def test_there_is_no_database_boundary_and_mcp_is_one_of_four():
    assert {b.value for b in Boundary} == {"retrieval", "backend", "model", "mcp"}


def test_deadline_and_sizes_are_bounded():
    for bad in (0, -1, 121):
        with pytest.raises(ValidationError):
            contract(deadline_seconds=bad)
    with pytest.raises(ValidationError):
        contract(max_output_bytes=10)


def test_unknown_fields_and_versions_are_refused():
    with pytest.raises(ValidationError):
        contract(provider="gemini")
    data = contract().model_dump(mode="json")
    data["schema_version"] = 2
    with pytest.raises(ValidationError):
        ToolContract.model_validate(data)


def test_contract_is_immutable():
    c = contract()
    with pytest.raises(ValidationError):
        c.name = "other_tool"  # type: ignore[misc]


def test_the_model_sees_no_permissions():
    seen = contract(required_permissions=("content:read",)).describe_for_model()
    assert set(seen) == {"name", "description", "input_schema"}


def test_contract_round_trips_through_json():
    c = contract(
        side_effect=SideEffect.PROPOSE,
        idempotency=Idempotency.KEYED,
        retry=RetryPolicy(classification=RetryClass.TRANSIENT, max_retries=2),
        boundaries=(Boundary.BACKEND, Boundary.MODEL),
        required_permissions=("progress:propose",),
    )
    again = ToolContract.model_validate_json(c.model_dump_json())
    assert again == c
    json.dumps(c.model_dump(mode="json"))  # nothing a JSON file can't hold


# --- CallerContext and AgentTask ---------------------------------------------


def test_context_identifiers_are_validated_and_unique():
    with pytest.raises(ValidationError):
        ctx(tenant_id="bad id with spaces")
    with pytest.raises(ValidationError):
        ctx(permissions=("a:b", "a:b"))
    assert ctx(denied_assets=None).denied_assets is None  # distinct from "none denied"
    assert ctx().official_attempt_active is False


def test_a_task_has_every_part_the_runtime_needs():
    t = task()
    assert t.budget.max_steps >= 1 and t.budget.max_tokens > 0
    assert t.evidence.require_sources and t.evidence.abstain_if_insufficient
    assert t.execution_policy.doom_loop_repeat_limit >= 2
    assert t.completion.require_final_answer


def test_task_tools_are_unique_names_and_required_tools_are_allowed():
    with pytest.raises(ValidationError):
        task(allowed_tools=("search_course_material", "search_course_material"))
    with pytest.raises(ValidationError):
        task(allowed_tools=("Bad-Name",))
    with pytest.raises(ValidationError, match="allowed"):
        task(completion=CompletionCriteria(required_tools=("create_study_plan",)))


def test_requiring_sources_needs_a_minimum():
    with pytest.raises(ValidationError):
        EvidenceRequirements(require_sources=True, min_sources=0)
    assert EvidenceRequirements(require_sources=False, min_sources=0).min_sources == 0


@pytest.mark.parametrize(
    "bad",
    [
        {"budget": {"max_steps": 0}},
        {"budget": {"max_steps": 51}},
        {"budget": {"max_tokens": 0}},
        {"budget": {"max_cost_usd": -1}},
        {"deadline_epoch": 0},
        {"goal": ""},
        {"task_type": "Bad Type"},
        {"execution_policy": {"doom_loop_repeat_limit": 1}},
    ],
)
def test_a_task_with_unbounded_or_invalid_limits_is_refused(bad):
    with pytest.raises(ValidationError):
        task(**bad)


def test_the_task_digest_follows_its_content():
    assert task().digest() == task().digest()
    assert task().digest() != task(goal="Something else").digest()


def test_the_exam_flag_does_not_change_which_task_a_checkpoint_belongs_to():
    exam = task(context=ctx(official_attempt_active=True))
    assert exam.digest() == task().digest()
    assert task(context=ctx(user_id="u2")).digest() != task().digest()


def test_task_round_trips_through_json_and_carries_no_sets():
    t = task(
        deadline_epoch=1_900_000_000.0,
        context=ctx(denied_assets=("a1", "a2"), official_attempt_active=False),
    )
    text = t.model_dump_json()
    assert AgentTask.model_validate_json(text) == t
    assert isinstance(json.loads(text)["context"]["denied_assets"], list)


# --- ExecutionCheckpoint -----------------------------------------------------


def call(**over):
    base = dict(call_id="c1", step=1, kind=CallKind.TOOL, name="x_tool", args_digest="0" * 64)
    base.update(over)
    return CallRecord(**base)


def test_a_new_checkpoint_is_pending_and_empty():
    cp = checkpoint()
    assert cp.state is ExecutionState.PENDING
    assert cp.step == 0 and cp.calls == () and cp.retry_count == 0
    assert cp.budget_consumed.tokens == 0
    assert cp.cancellation.requested is False


def test_a_run_that_did_not_complete_says_why():
    with pytest.raises(ValidationError, match="says why"):
        checkpoint(state=ExecutionState.FAILED)
    assert checkpoint(state=ExecutionState.FAILED, outcome_code="MODEL_UNAVAILABLE").outcome_code
    assert checkpoint(state=ExecutionState.COMPLETED).state is ExecutionState.COMPLETED


def test_a_run_still_going_has_no_outcome():
    with pytest.raises(ValidationError):
        checkpoint(state=ExecutionState.RUNNING, outcome_code="X")


def test_calls_cannot_come_from_the_future_or_repeat_ids():
    with pytest.raises(ValidationError, match="not happened"):
        checkpoint(step=1, calls=(call(step=2),))
    with pytest.raises(ValidationError, match="unique"):
        checkpoint(step=1, calls=(call(), call()))


def test_a_call_may_only_point_at_a_listed_result():
    with pytest.raises(ValidationError, match="result"):
        checkpoint(step=1, calls=(call(result_ref="r1"),))
    ref = ResultRef(ref_id="r1", call_id="c1", step=1, sha256="a" * 64)
    assert checkpoint(step=1, calls=(call(result_ref="r1"),), results=(ref,))


def test_call_records_hold_digests_not_arguments():
    assert "arguments" not in CallRecord.model_fields
    assert "args" not in CallRecord.model_fields
    with pytest.raises(ValidationError):
        call(args_digest="the student's text")


def test_cancellation_has_a_reason_and_only_a_cancellation_does():
    with pytest.raises(ValidationError):
        checkpoint(cancellation={"requested": True})
    with pytest.raises(ValidationError):
        checkpoint(cancellation={"requested": False, "reason": "user"})
    cp = checkpoint(
        state=ExecutionState.CANCELLED,
        outcome_code="CANCELLED",
        cancellation={"requested": True, "reason": "user", "at_step": 2},
        step=2,
    )
    assert cp.cancellation.reason == "user"
    with pytest.raises(ValidationError, match="cancelled run"):
        checkpoint(
            state=ExecutionState.COMPLETED,
            cancellation={"requested": True, "reason": "user", "at_step": 1},
            step=1,
        )


def test_checkpoint_round_trips_through_json():
    ref = ResultRef(ref_id="r1", call_id="c1", step=1, sha256="a" * 64)
    cp = checkpoint(
        step=1,
        state=ExecutionState.RUNNING,
        calls=(call(result_ref="r1"),),
        results=(ref,),
        retry_count=1,
    )
    assert ExecutionCheckpoint.model_validate_json(cp.model_dump_json()) == cp


def test_state_machine_has_no_way_out_of_an_end_state():
    for state in TERMINAL_STATES:
        assert ALLOWED_TRANSITIONS.get(state, frozenset()) == frozenset()
    assert can_transition(ExecutionState.PENDING, ExecutionState.RUNNING)
    assert can_transition(ExecutionState.PENDING, ExecutionState.REFUSED)
    assert not can_transition(ExecutionState.PENDING, ExecutionState.COMPLETED)
    assert can_transition(ExecutionState.RUNNING, ExecutionState.CANCELLED)
    assert not can_transition(ExecutionState.RUNNING, ExecutionState.REFUSED)
    assert not can_transition(ExecutionState.COMPLETED, ExecutionState.RUNNING)
    assert not can_transition(ExecutionState.CANCELLED, ExecutionState.RUNNING)


def test_canonical_json_is_stable_and_refuses_nan():
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})
    assert digest({"a": 1}) == digest({"a": 1}) != digest({"a": 2})
    with pytest.raises(ValueError):
        canonical_json({"a": float("nan")})


# --- Portability -------------------------------------------------------------

PORTABLE = ["schema.py", "contracts.py"]
ALLOWED_ROOTS = {
    "__future__",
    "hashlib",
    "json",
    "math",
    "re",
    "enum",
    "typing",
    "pydantic",
    "app",
}


def test_the_contracts_import_nothing_server_or_provider_shaped():
    """A device implementation needs only these files, pydantic and the
    standard library: no web framework, database, gateway or provider."""
    root = Path(__file__).resolve().parents[2] / "app" / "core" / "agentic"
    for name in PORTABLE:
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                assert module.split(".")[0] in ALLOWED_ROOTS, (name, module)
                if module.startswith("app"):
                    assert module in {"app.core.agentic.schema"}, (name, module)
    text = "".join((root / n).read_text(encoding="utf-8").lower() for n in PORTABLE)
    for provider in ("gemini", "fireworks", "openai", "anthropic"):
        assert provider not in text
