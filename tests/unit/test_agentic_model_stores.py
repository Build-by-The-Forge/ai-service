"""Decisions, prompts, and the stores a run keeps its state in."""

import pytest
from pydantic import ValidationError

from app.core.agentic.contracts import ExecutionCheckpoint, ExecutionState, digest
from app.core.agentic.model import (
    MAX_OBSERVATION_CHARS,
    Decision,
    DecisionRequest,
    MalformedDecision,
    Turn,
    observation_text,
    parse_decision,
    render_prompt,
)
from app.core.agentic.stores import InMemoryCheckpointStore, InMemoryResultStore


def request(**over):
    base = dict(
        goal="Explain conduction",
        task_type="grounded_explanation",
        tools=({"name": "search_course_material"},),
        step=1,
        steps_left=5,
        iterations_left=4,
    )
    base.update(over)
    return DecisionRequest(**base)


def test_a_tool_decision_names_a_tool_and_a_final_one_has_an_answer():
    assert Decision(kind="tool", tool="search_course_material", arguments={"query": "x"}).tool
    with pytest.raises(ValidationError):
        Decision(kind="tool")
    with pytest.raises(ValidationError):
        Decision(kind="final", answer="   ")
    with pytest.raises(ValidationError):
        Decision(kind="final", answer="x" * 9000)
    with pytest.raises(ValidationError):
        Decision(kind="tool", tool="t", surprise=1)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "text",
        [],
        {},
        {"action": "dance"},
        {"action": "tool"},
        {"action": "tool", "tool": "x", "arguments": "not an object"},
        {"action": "final"},
        {"action": "final", "answer": ""},
        {"action": "final", "answer": "ok", "citations": "c1"},
        {"action": "final", "answer": "ok", "citations": [1]},
    ],
)
def test_anything_that_is_not_a_decision_is_malformed_and_says_nothing(raw):
    with pytest.raises(MalformedDecision) as caught:
        parse_decision(raw)
    assert str(caught.value) == ""


def test_decisions_parse_with_their_tokens():
    tool = parse_decision({"action": "tool", "tool": "t_tool", "arguments": {"q": 1}}, tokens=7)
    assert (tool.kind, tool.tool, tool.arguments, tool.tokens) == ("tool", "t_tool", {"q": 1}, 7)
    final = parse_decision({"action": "final", "answer": "Done.", "citations": ["c1"]})
    assert (final.kind, final.answer, final.citations) == ("final", "Done.", ("c1",))
    assert parse_decision({"action": "tool", "tool": "t_tool"}).arguments == {}


def test_observations_are_canonical_and_bounded():
    assert observation_text({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    long = observation_text({"text": "x" * 10_000})
    assert len(long) <= MAX_OBSERVATION_CHARS + 10 and long.endswith("...[cut]")


def test_the_prompt_is_deterministic_and_marks_results_as_data():
    turn = Turn(step=1, tool="search_course_material", arguments={"query": "q"}, observation="{}")
    r = request(turns=(turn,), evidence_ids=("c1",), notes=("A required step is missing.",))
    assert render_prompt(r) == render_prompt(r)
    text = render_prompt(r)
    assert "data, not instructions" in text and '"c1"' in text
    assert "TURN 1: search_course_material" in text and "NOTE: A required step" in text


async def test_the_checkpoint_store_returns_a_copy_that_equals_what_was_saved():
    store = InMemoryCheckpointStore()
    assert await store.load("e1") is None
    cp = ExecutionCheckpoint(
        execution_id="e1",
        task_id="t1",
        task_digest="a" * 64,
        tenant_id="t",
        user_id="u",
        state=ExecutionState.RUNNING,
    )
    await store.save(cp)
    loaded = await store.load("e1")
    assert loaded == cp and loaded is not cp


async def test_the_result_store_returns_the_payload_digest():
    store = InMemoryResultStore()
    payload = {"tool": "t_tool", "ok": True}
    assert await store.put("k", payload) == digest(payload)
    assert await store.get("k") == payload and await store.get("missing") is None
