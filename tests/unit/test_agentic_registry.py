"""The registry: every call, whoever makes it, takes one checked path and leaves
one audit record."""

import asyncio
import logging

import pytest

from app.core.agentic.contracts import (
    Boundary,
    CallerContext,
    Idempotency,
    RetryClass,
    RetryPolicy,
    SideEffect,
    ToolContract,
)
from app.core.agentic.ports import Ports
from app.core.agentic.registry import (
    InMemoryAuditSink,
    RegistryPolicy,
    RunControl,
    ToolCall,
    ToolFailure,
    ToolRegistry,
)

IN = {
    "type": "object",
    "properties": {"q": {"type": "string", "maxLength": 20}, "tenant_id": {"type": "string"}},
    "required": ["q"],
    "additionalProperties": False,
}
OUT = {
    "type": "object",
    "properties": {"a": {"type": "string"}},
    "required": ["a"],
    "additionalProperties": False,
}


def contract(name="echo_tool", **over):
    base = dict(
        name=name,
        version="1.0.0",
        description="A test tool that echoes its query.",
        input_schema=IN,
        output_schema=OUT,
    )
    base.update(over)
    return ToolContract(**base)


def keyed(name="keyed_tool", **over):
    return contract(name, side_effect=SideEffect.PROPOSE, idempotency=Idempotency.KEYED, **over)


def ctx(**over):
    base = dict(
        tenant_id="t1",
        user_id="u1",
        permissions=("content:read",),
        capabilities=("agentic_workflows",),
        denied_assets=(),
    )
    base.update(over)
    return CallerContext(**base)


async def echo(call: ToolCall):
    return {"a": call.args["q"]}


def make(*tools, ports=None, policy=None):
    registry = ToolRegistry(ports or Ports(), policy=policy)
    for c, h in tools:
        registry.register(c, h)
    return registry


async def run(registry, name, args, context=None, **kw):
    kw.setdefault("call_id", "c1")
    kw.setdefault("execution_id", "e1")
    return await registry.execute(name, args, context or ctx(), **kw)


def last(registry):
    assert isinstance(registry.audit, InMemoryAuditSink)
    return registry.audit.records[-1]


# --- registration and lookup -------------------------------------------------


def test_duplicate_and_synchronous_handlers_are_refused():
    registry = make((contract(), echo))
    with pytest.raises(ValueError):
        registry.register(contract(), echo)
    with pytest.raises(TypeError):
        registry.register(contract("other_tool"), lambda call: {})  # type: ignore[arg-type]


async def test_an_unknown_tool_is_refused_and_audited():
    registry = make()
    out = await run(registry, "no_such_tool", {})
    assert not out.ok and out.error_code == "TOOL_NOT_FOUND"
    assert last(registry).status == "denied" and last(registry).tool_version is None


async def test_a_tool_outside_the_task_allow_list_is_refused():
    registry = make((contract(), echo))
    out = await run(registry, "echo_tool", {"q": "x"}, allowed=("other_tool",))
    assert out.error_code == "TOOL_NOT_ALLOWED"
    ok = await run(registry, "echo_tool", {"q": "x"}, allowed=("echo_tool",))
    assert ok.ok and ok.data == {"a": "x"}


# --- authority ----------------------------------------------------------------


async def test_a_missing_permission_or_capability_is_refused():
    registry = make(
        (contract(required_permissions=("progress:read",)), echo),
        (contract("cap_tool", required_capabilities=("model_tier:high",)), echo),
    )
    assert (await run(registry, "echo_tool", {"q": "x"})).error_code == "PERMISSION_DENIED"
    assert (await run(registry, "cap_tool", {"q": "x"})).error_code == "CAPABILITY_DENIED"
    allowed = ctx(permissions=("progress:read",))
    assert (await run(registry, "echo_tool", {"q": "x"}, allowed)).ok


async def test_no_tool_runs_during_an_official_attempt():
    registry = make((contract(), echo))
    out = await run(registry, "echo_tool", {"q": "x"}, ctx(official_attempt_active=True))
    assert out.error_code == "EXAM_MODE_ACTIVE"
    assert registry.available(ctx(official_attempt_active=True)) == []


async def test_a_write_through_backend_is_refused_unless_the_registry_allows_it():
    write = contract(
        "write_tool",
        side_effect=SideEffect.WRITE_THROUGH_BACKEND,
        idempotency=Idempotency.KEYED,
        boundaries=(Boundary.BACKEND,),
    )
    calls = []

    async def handler(call):
        calls.append(1)
        return {"a": "done"}

    registry = make((write, handler))
    assert (await run(registry, "write_tool", {"q": "x"})).error_code == "SIDE_EFFECT_DISALLOWED"
    assert calls == []
    allowing = make((write, handler), policy=RegistryPolicy(allow_write_through_backend=True))
    assert (await run(allowing, "write_tool", {"q": "x"})).ok


async def test_proposals_can_be_switched_off():
    registry = make((keyed(), echo), policy=RegistryPolicy(allow_proposals=False))
    assert (await run(registry, "keyed_tool", {"q": "x"})).error_code == "SIDE_EFFECT_DISALLOWED"


async def test_available_lists_only_what_the_caller_may_use():
    registry = make(
        (contract(), echo),
        (contract("secret_tool", required_permissions=("admin:all",)), echo),
    )
    assert [c.name for c in registry.available(ctx())] == ["echo_tool"]
    assert registry.available(ctx(), allowed=("secret_tool",)) == []


async def test_arguments_naming_another_tenant_or_user_are_refused():
    registry = make((contract(), echo))
    other = await run(registry, "echo_tool", {"q": "x", "tenant_id": "t2"})
    assert other.error_code == "TENANT_MISMATCH"
    same = await run(registry, "echo_tool", {"q": "x", "tenant_id": "t1"})
    assert same.ok


# --- arguments and results -----------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [{}, {"q": 5}, {"q": "x" * 21}, {"q": "x", "extra": 1}, {"q": float("nan")}],
)
async def test_arguments_are_checked_against_the_schema(args):
    registry = make((contract(), echo))
    out = await run(registry, "echo_tool", args)
    assert out.error_code == "INVALID_ARGUMENTS" and not out.retryable
    assert last(registry).status == "invalid"


async def test_oversized_arguments_are_refused_before_the_handler_runs():
    calls = []

    async def handler(call):
        calls.append(1)
        return {"a": "x"}

    registry = make((contract(max_input_bytes=64), handler))
    out = await run(registry, "echo_tool", {"q": "x", "tenant_id": "t" * 100})
    assert out.error_code == "INVALID_ARGUMENTS" and calls == []


@pytest.mark.parametrize(
    "result",
    [{"a": 1}, {}, {"a": "x", "extra": 1}, "text", ["a"], {"a": "x" * 5000}],
)
async def test_results_are_checked_against_the_output_schema_and_size(result):
    async def handler(call):
        return result

    registry = make((contract(max_output_bytes=200), handler))
    out = await run(registry, "echo_tool", {"q": "x"})
    assert out.error_code == "TOOL_OUTPUT_INVALID" and out.data is None


# --- failures never leak ------------------------------------------------------


async def test_an_exception_reaches_the_model_as_a_fixed_message_only(caplog):
    secret = "the student's private essay on thermodynamics"

    async def handler(call):
        raise RuntimeError(secret)

    registry = make((contract(), handler))
    with caplog.at_level(logging.DEBUG):
        out = await run(registry, "echo_tool", {"q": "x"})
    assert out.error_code == "TOOL_FAILED"
    blob = repr(out.model_dump()) + repr(last(registry)) + caplog.text
    assert secret not in blob
    assert "RuntimeError" in caplog.text  # type and place are logged


async def test_a_tool_failure_keeps_its_code_and_unknown_codes_become_tool_failed():
    async def scoped(call):
        raise ToolFailure("MISSING_SCOPE")

    async def odd(call):
        raise ToolFailure("SOMETHING_ELSE")

    registry = make((contract(), scoped), (contract("odd_tool"), odd))
    assert (await run(registry, "echo_tool", {"q": "x"})).error_code == "MISSING_SCOPE"
    assert (await run(registry, "odd_tool", {"q": "x"})).error_code == "TOOL_FAILED"


async def test_failure_shape_is_code_retry_flag_and_fixed_message():
    registry = make()
    shape = (await run(registry, "no_such_tool", {})).failure()
    assert shape == {
        "ok": False,
        "error": {
            "code": "TOOL_NOT_FOUND",
            "retryable": False,
            "message": "The tool is not available.",
        },
    }


# --- timeouts, retries ----------------------------------------------------------


async def test_a_slow_tool_times_out_at_its_own_deadline():
    async def slow(call):
        await asyncio.sleep(5)
        return {"a": "x"}

    registry = make((contract(deadline_seconds=0.05), slow))
    out = await run(registry, "echo_tool", {"q": "x"})
    assert out.error_code == "TOOL_TIMEOUT" and out.retryable
    assert last(registry).status == "timeout"


async def test_the_runs_deadline_tightens_the_tools():
    async def slow(call):
        await asyncio.sleep(5)
        return {"a": "x"}

    registry = make((contract(deadline_seconds=60), slow))
    control = RunControl(deadline=RunControl().clock() + 0.05)
    started = asyncio.get_running_loop().time()
    out = await run(registry, "echo_tool", {"q": "x"}, control=control)
    assert out.error_code == "DEADLINE_EXPIRED" and not out.effect_known
    assert asyncio.get_running_loop().time() - started < 2


async def test_transient_failures_are_retried_up_to_the_contract_and_then_succeed():
    attempts = []

    async def flaky(call):
        attempts.append(1)
        if len(attempts) < 3:
            raise ToolFailure("TOOL_UNAVAILABLE")
        return {"a": "ok"}

    retry = RetryPolicy(classification=RetryClass.TRANSIENT, max_retries=2)
    registry = make((contract(retry=retry), flaky))
    out = await run(registry, "echo_tool", {"q": "x"})
    assert out.ok and out.retries == 2 and len(attempts) == 3
    assert last(registry).retries == 2


async def test_retries_stop_at_the_limit_and_the_failure_says_it_may_be_retried_later():
    attempts = []

    async def down(call):
        attempts.append(1)
        raise ToolFailure("TOOL_UNAVAILABLE")

    retry = RetryPolicy(classification=RetryClass.TRANSIENT, max_retries=1)
    registry = make((contract(retry=retry), down))
    out = await run(registry, "echo_tool", {"q": "x"})
    assert not out.ok and out.retryable and out.retries == 1 and len(attempts) == 2


async def test_a_tool_that_is_never_retried_is_not():
    attempts = []

    async def down(call):
        attempts.append(1)
        raise ToolFailure("TOOL_UNAVAILABLE")

    registry = make((contract(), down))
    await run(registry, "echo_tool", {"q": "x"})
    assert len(attempts) == 1


async def test_a_non_transient_failure_is_not_retried_even_when_retries_are_allowed():
    attempts = []

    async def broken(call):
        attempts.append(1)
        raise RuntimeError("x")

    retry = RetryPolicy(classification=RetryClass.TRANSIENT, max_retries=3)
    registry = make((contract(retry=retry), broken))
    await run(registry, "echo_tool", {"q": "x"})
    assert len(attempts) == 1


# --- idempotency ----------------------------------------------------------------


async def test_a_repeat_of_a_keyed_call_returns_the_first_outcome_with_no_second_effect():
    effects = []

    async def handler(call):
        effects.append(call.idempotency_key)
        return {"a": f"proposal-{len(effects)}"}

    registry = make((keyed(), handler))
    first = await run(registry, "keyed_tool", {"q": "x"}, call_id="c1")
    again = await run(registry, "keyed_tool", {"q": "x"}, call_id="c2")
    assert first.ok and again.ok and not first.replayed and again.replayed
    assert again.data == first.data and again.call_id == "c2"
    assert len(effects) == 1
    assert [r.status for r in registry.audit.records] == ["ok", "replayed"]  # type: ignore[attr-defined]


async def test_the_derived_key_is_stable_and_an_explicit_key_is_used():
    seen = []

    async def handler(call):
        seen.append(call.idempotency_key)
        return {"a": "x"}

    registry = make((keyed(), handler))
    await run(registry, "keyed_tool", {"q": "x"}, execution_id="e1")
    await run(registry, "keyed_tool", {"q": "y"}, execution_id="e1")
    await run(registry, "keyed_tool", {"q": "x"}, execution_id="e2")
    await run(registry, "keyed_tool", {"q": "z"}, idempotency_key="my-key")
    assert len(set(seen[:3])) == 3 and seen[3] == "my-key"
    fresh = make((keyed(), handler))
    await run(fresh, "keyed_tool", {"q": "x"}, execution_id="e1")
    assert seen[4] == seen[0]  # the same call derives the same key anywhere


async def test_a_key_reused_for_different_arguments_is_a_conflict():
    registry = make((keyed(), echo))
    await run(registry, "keyed_tool", {"q": "x"}, idempotency_key="k")
    out = await run(registry, "keyed_tool", {"q": "other"}, idempotency_key="k")
    assert out.error_code == "IDEMPOTENCY_CONFLICT"


async def test_one_students_key_never_answers_for_another():
    effects = []

    async def handler(call):
        effects.append(call.context.user_id)
        return {"a": "x"}

    registry = make((keyed(), handler))
    await run(registry, "keyed_tool", {"q": "x"}, ctx(user_id="u1"), idempotency_key="k")
    other = await run(registry, "keyed_tool", {"q": "x"}, ctx(user_id="u2"), idempotency_key="k")
    assert not other.replayed and effects == ["u1", "u2"]


async def test_concurrent_calls_with_one_key_run_once():
    effects = []

    async def handler(call):
        effects.append(1)
        await asyncio.sleep(0.05)
        return {"a": "x"}

    registry = make((keyed(), handler))
    results = await asyncio.gather(
        *[run(registry, "keyed_tool", {"q": "x"}, call_id=f"c{i}") for i in range(5)]
    )
    assert len(effects) == 1 and sum(r.replayed for r in results) == 4


async def test_a_timed_out_keyed_call_is_not_recorded_so_a_repeat_settles_it():
    attempts = []

    async def handler(call):
        attempts.append(call.idempotency_key)
        if len(attempts) == 1:
            await asyncio.sleep(5)
        return {"a": "x"}

    registry = make((keyed(deadline_seconds=0.05), handler))
    first = await run(registry, "keyed_tool", {"q": "x"})
    second = await run(registry, "keyed_tool", {"q": "x"})
    assert first.error_code == "TOOL_TIMEOUT" and second.ok and not second.replayed
    assert attempts[0] == attempts[1]  # the same key, so the backend can settle it


async def test_a_read_is_not_cached():
    effects = []

    async def handler(call):
        effects.append(1)
        return {"a": "x"}

    registry = make((contract(), handler))
    await run(registry, "echo_tool", {"q": "x"})
    await run(registry, "echo_tool", {"q": "x"})
    assert len(effects) == 2


# --- cancellation and deadline ----------------------------------------------------


async def test_a_cancelled_run_calls_no_tool():
    calls = []

    async def handler(call):
        calls.append(1)
        return {"a": "x"}

    registry = make((contract(), handler))
    control = RunControl()
    control.cancel("user")
    out = await run(registry, "echo_tool", {"q": "x"}, control=control)
    assert out.error_code == "EXECUTION_CANCELLED" and calls == []
    assert last(registry).status == "cancelled"


async def test_a_run_past_its_deadline_calls_no_tool():
    registry = make((contract(), echo))
    control = RunControl(deadline=RunControl().clock() - 1)
    out = await run(registry, "echo_tool", {"q": "x"}, control=control)
    assert out.error_code == "DEADLINE_EXPIRED"


async def test_cancelling_mid_call_stops_the_handler_and_marks_the_effect_unknown():
    started = asyncio.Event()
    finished = []

    async def handler(call):
        started.set()
        await asyncio.sleep(5)
        finished.append(1)
        return {"a": "x"}

    registry = make((keyed(deadline_seconds=30), handler))
    control = RunControl()
    task = asyncio.ensure_future(
        run(registry, "keyed_tool", {"q": "x"}, control=control, idempotency_key="k")
    )
    await started.wait()
    control.cancel("parent")
    out = await asyncio.wait_for(task, 2)
    assert out.error_code == "EXECUTION_CANCELLED" and not out.effect_known
    assert finished == []
    # Not recorded: a repeat with the same key is what settles it.
    again = await run(registry, "keyed_tool", {"q": "x"}, idempotency_key="k")
    assert again.error_code != "IDEMPOTENCY_CONFLICT" and not again.replayed


async def test_cancelling_the_caller_cancels_the_handler():
    started = asyncio.Event()
    cancelled = []

    async def handler(call):
        started.set()
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise
        return {"a": "x"}

    registry = make((contract(deadline_seconds=30), handler))
    task = asyncio.ensure_future(run(registry, "echo_tool", {"q": "x"}))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert cancelled == [1]


# --- audit and boundaries -----------------------------------------------------------


async def test_every_outcome_writes_exactly_one_audit_record_without_content():
    secret = "private student text"

    async def handler(call):
        return {"a": secret}

    registry = make((contract(), handler), (keyed(), echo))
    await run(registry, "echo_tool", {"q": secret})
    await run(registry, "echo_tool", {})
    await run(registry, "no_such_tool", {"q": secret})
    await run(registry, "keyed_tool", {"q": "x"}, ctx(permissions=()))
    records = registry.audit.records  # type: ignore[attr-defined]
    assert len(records) == 4
    assert [r.status for r in records] == ["ok", "invalid", "denied", "ok"]
    for record in records:
        assert secret not in record.model_dump_json()
        assert record.tenant_id == "t1" and record.user_id == "u1"
        assert len(record.args_digest) == 64
    ok = records[0]
    assert ok.input_bytes > 0 and ok.output_bytes > 0 and ok.tool_version == "1.0.0"
    assert records[3].idempotency_key  # the keyed call's key is audited


async def test_an_audit_sink_failure_does_not_change_the_outcome(caplog):
    class Broken:
        def record(self, record):
            raise OSError("disk full: secret path")

    registry = ToolRegistry(Ports(), audit=Broken())
    registry.register(contract(), echo)
    out = await run(registry, "echo_tool", {"q": "x"})
    assert out.ok and "secret path" not in caplog.text


async def test_a_handler_reaching_for_an_undeclared_door_fails_closed():
    class Retrieval:
        async def gather(self, query, *, tenant_id, denied_assets):
            raise AssertionError("must not be reached")

    async def greedy(call):
        _ = call.services.retrieval
        return {"a": "x"}

    registry = make((contract(), greedy), ports=Ports(retrieval=Retrieval()))
    out = await run(registry, "echo_tool", {"q": "x"})
    assert out.error_code == "TOOL_FAILED"


async def test_nested_model_tokens_are_charged_to_the_call():
    class Model:
        async def complete(self, prompt):
            return "answer " * 10

        async def complete_json(self, prompt, *, kind, **hints):
            return {}

    async def handler(call):
        return {"a": await call.services.model.complete("p" * 40)}

    registry = make((contract(boundaries=(Boundary.MODEL,)), handler), ports=Ports(model=Model()))
    out = await run(registry, "echo_tool", {"q": "x"})
    assert out.model_tokens == 10 + 18  # 40 chars and 70 chars, four per token, rounded up
    assert last(registry).model_tokens == out.model_tokens
