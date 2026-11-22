"""The agent evaluation harness: a versioned synthetic suite, deterministic
scoring, traces and replay, and negative controls that prove it measures."""

import json
import re
from pathlib import Path

import pytest

from app.core.agentic.evalkit.harness import run_suite, run_task
from app.core.agentic.evalkit.policies import POLICIES, reference
from app.core.agentic.evalkit.scoring import Facts, score
from app.core.agentic.evalkit.suite import (
    CATEGORIES,
    DEFAULT_SUITE,
    Expect,
    load_suite,
)
from app.core.agentic.evalkit.trace import ReplayAgentModel, Trace
from app.core.agentic.model import Decision

BASELINE = DEFAULT_SUITE.parent / "baseline.json"
SUITE = load_suite()


# --- the suite --------------------------------------------------------------------------------


def test_the_suite_is_versioned_and_covers_every_category():
    assert (SUITE.version, SUITE.name) == (1, "agent-eval-synthetic-v1")
    by = {c: [t for t in SUITE.tasks if t.category == c] for c in CATEGORIES}
    assert set(by) == set(CATEGORIES) and all(len(v) >= 3 for v in by.values())
    assert len({t.id for t in SUITE.tasks}) == len(SUITE.tasks) == 24


def test_the_suite_file_is_pinned_by_its_digest():
    pinned = json.loads(BASELINE.read_text(encoding="utf-8"))["suite_sha256"]
    assert (
        SUITE.sha256 == pinned
    ), "tasks.json changed: re-run scripts/run_agent_eval.py --write-baseline"


def test_the_suite_is_synthetic_and_original():
    raw = DEFAULT_SUITE.read_text(encoding="utf-8")
    assert "synthetic" in raw.lower()
    # no real institution, person or course code
    assert not re.search(r"\b(FUTA|CSC|PHY|MEE|CHM)\s?\d{3}\b", raw)
    assert not re.search(r"[\w.]+@[\w.]+\.\w+", raw)
    assert {p.tenant_id for p in SUITE.corpus} == {"eval-tenant"}


def test_every_task_has_a_script_and_a_stated_expectation():
    for task in SUITE.tasks:
        assert task.script and task.expect.state
        assert set(task.expect.tools) <= {"search_course_material", *task.allowed_tools} | set(
            task.allowed_tools
        )


# --- scoring ------------------------------------------------------------------------------------


def facts(**over):
    base = dict(
        state="completed",
        code=None,
        attempted_tools=["search_course_material"],
        calls=[{"tool": "search_course_material", "arguments": {"query": "q", "max_passages": 2}}],
        proposals=0,
        cited=["c1"],
        steps=2,
    )
    base.update(over)
    return Facts(**base)


def expect(**over):
    base = dict(state="completed", tools=("search_course_material",), cited=("c1",), max_steps=2)
    base.update(over)
    return Expect(**base)


def test_a_run_that_matches_its_expectation_passes_with_every_component_at_one():
    scored = score(facts(), expect())
    assert scored.passed and scored.mean == 1.0


@pytest.mark.parametrize(
    "over, component",
    [
        ({"state": "failed"}, "state"),
        ({"code": "DOOM_LOOP"}, "code"),
        ({"attempted_tools": ["explain_concept"]}, "tool_selection"),
        (
            {"calls": [{"tool": "search_course_material", "arguments": {"query": "q"}}]},
            "argument_accuracy",
        ),
        ({"proposals": 1}, "effects"),
        ({"cited": []}, "citations"),
        ({"steps": 5}, "efficiency"),
    ],
)
def test_each_component_catches_its_own_failure(over, component):
    exp = expect(args=({"tool": "search_course_material", "subset": {"max_passages": 2}},))
    scored = score(facts(**over), exp)
    assert getattr(scored, component) < 1.0 and not scored.passed
    others = {k: v for k, v in scored.model_dump().items() if k != component}
    assert all(v == 1.0 for v in others.values()), others


def test_attempting_a_forbidden_tool_fails_even_if_it_was_refused():
    scored = score(
        facts(attempted_tools=["search_course_material", "update_learning_progress"]),
        expect(
            tools=("search_course_material", "update_learning_progress"),
            forbidden_tools=("update_learning_progress",),
        ),
    )
    assert scored.forbidden == 0.0 and not scored.passed


def test_tool_selection_gives_partial_credit_for_the_right_set_in_the_wrong_order():
    two = ("search_course_material", "explain_concept")
    exact = score(facts(attempted_tools=list(two)), expect(tools=two)).tool_selection
    swapped = score(facts(attempted_tools=list(reversed(two))), expect(tools=two)).tool_selection
    wrong = score(facts(attempted_tools=["evaluate_response"]), expect(tools=two)).tool_selection
    assert exact == 1.0 and 0 < swapped < 1.0 and wrong == 0.0


# --- the reference policy ---------------------------------------------------------------------------


async def test_the_reference_policy_passes_every_task():
    report, _ = await run_suite(SUITE, reference, "reference")
    failed = [r.task_id for r in report.results if not r.passed]
    assert failed == [] and report.pass_rate == 1.0 and report.mean == 1.0
    assert set(report.by_category) == set(CATEGORIES)


async def test_each_task_ends_in_the_state_its_category_is_about():
    report, _ = await run_suite(SUITE, reference, "reference")
    states = {r.task_id: (r.facts.state, r.facts.code) for r in report.results}
    assert states["loop-01"] == ("loop_detected", "DOOM_LOOP")
    assert states["loop-02"] == ("loop_detected", "NO_PROGRESS")
    assert states["abs-01"] == ("abstained", "INSUFFICIENT_EVIDENCE")
    assert states["bud-02"] == ("budget_exhausted", "ITERATION_BUDGET")
    assert states["can-01"] == ("cancelled", "CANCELLED")
    assert states["perm-02"] == ("refused", "EXAM_MODE_ACTIVE")
    assert states["perm-03"] == ("refused", "PERMISSION_DENIED")
    assert states["rep-03"] == ("completed", None)


async def test_the_replay_and_idempotency_tasks_have_exactly_one_effect():
    report, _ = await run_suite(SUITE, reference, "reference")
    proposals = {
        r.task_id: r.facts.proposals for r in report.results if r.task_id.startswith("rep-")
    }
    assert proposals == {"rep-01": 1, "rep-02": 0, "rep-03": 1}


# --- negative controls: the scoring can tell a bad model from a good one ------------------------------------


@pytest.mark.parametrize("name", ["naive", "looping", "overreach"])
async def test_a_bad_policy_scores_clearly_below_the_reference(name):
    report, _ = await run_suite(SUITE, POLICIES[name], name)
    assert report.pass_rate <= 0.4 and report.mean < 0.8


async def test_the_overreaching_policy_is_caught_on_tool_selection_and_citations():
    report, _ = await run_suite(SUITE, POLICIES["overreach"], "overreach")
    sel = next(r for r in report.results if r.task_id == "sel-02")
    assert "update_learning_progress" in sel.facts.attempted_tools
    assert sel.scores.tool_selection < 1.0 and not sel.passed
    assert sel.facts.state == "failed" and sel.facts.code == "INVENTED_SOURCE"


async def test_the_naive_policy_cannot_pass_a_grounded_task_by_inventing_a_score():
    report, _ = await run_suite(SUITE, POLICIES["naive"], "naive")
    sel = next(r for r in report.results if r.task_id == "sel-01")
    assert sel.facts.code in ("INSUFFICIENT_EVIDENCE", "UNSUPPORTED_FIGURE") and not sel.passed


# --- determinism, traces and replay ----------------------------------------------------------------------------


async def test_the_same_run_twice_gives_the_same_report():
    a, _ = await run_suite(SUITE, reference, "reference")
    b, _ = await run_suite(SUITE, reference, "reference")
    assert a.digest() == b.digest()


async def test_the_baseline_is_reproduced_exactly():
    recorded = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert recorded["suite_sha256"] == SUITE.sha256
    for row in recorded["policies"]:
        report, _ = await run_suite(SUITE, POLICIES[row["policy"]], row["policy"])
        assert report.digest() == row["report_digest"], row["policy"]
        assert (report.pass_rate, report.mean) == (row["pass_rate"], row["mean"])
    assert {r["policy"] for r in recorded["policies"]} == set(POLICIES)


async def test_a_recorded_trace_replays_to_the_same_result_without_the_original_model():
    report, traces = await run_suite(SUITE, reference, "reference")
    replayed, _ = await run_suite(SUITE, lambda task: ReplayAgentModel(traces[task.id]), "replay")
    assert [r.result_digest for r in replayed.results] == [r.result_digest for r in report.results]
    assert replayed.pass_rate == 1.0


async def test_a_trace_survives_a_round_trip_through_json():
    task = next(t for t in SUITE.tasks if t.id == "sel-02")
    _, trace, _ = await run_task(SUITE, task, reference)
    again = Trace.model_validate_json(trace.model_dump_json())
    assert again == trace and trace.suite_sha256 == SUITE.sha256
    assert [e["type"] for e in trace.events] == ["decision", "tool", "decision", "end"]


async def test_changing_a_recorded_decision_changes_the_score():
    task = next(t for t in SUITE.tasks if t.id == "sel-02")
    _, trace, _ = await run_task(SUITE, task, reference)
    for event in trace.events:
        if event["type"] == "decision" and event["decision"]["kind"] == "final":
            event["decision"]["citations"] = []
    outcome, _, _ = await run_task(SUITE, task, lambda t: ReplayAgentModel(trace))
    assert not outcome.passed and outcome.facts.state == "abstained"


async def test_a_trace_replays_a_resumed_run_by_step_not_by_call_count():
    task = next(t for t in SUITE.tasks if t.id == "rep-02")  # crashes, then resumes
    first, trace, _ = await run_task(SUITE, task, reference)
    again, _, _ = await run_task(SUITE, task, lambda t: ReplayAgentModel(trace))
    assert first.passed and again.passed and first.result_digest == again.result_digest


# --- plugging a model in --------------------------------------------------------------------------------------------


async def test_any_model_factory_plugs_in_and_the_harness_never_needs_a_provider():
    class Direct:
        """A model that always does the sensible thing for one task."""

        async def decide(self, request):
            if request.step == 1:
                return Decision(
                    kind="tool",
                    tool="search_course_material",
                    arguments={"query": "buoyant force displaced fluid"},
                    tokens=5,
                )
            return Decision(
                kind="final",
                answer="Buoyancy is the displaced fluid's weight.",
                citations=("c-fluid-3",),
                tokens=5,
            )

    task = next(t for t in SUITE.tasks if t.id == "sel-02")
    outcome, _, _ = await run_task(SUITE, task, lambda t: Direct())
    assert outcome.passed


def test_the_evalkit_imports_no_network_or_provider_code():
    import ast

    banned = {
        "httpx",
        "requests",
        "aiohttp",
        "socket",
        "subprocess",
        "openai",
        "google",
        "anthropic",
    }
    for path in Path("app/core/agentic/evalkit").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not {n.split(".")[0] for n in names} & banned, path.name
