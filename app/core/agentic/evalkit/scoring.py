"""Deterministic scoring of one task from what happened in the run.

No model grades anything. Each component is a number in [0, 1] computed from the
run's facts and the task's expectation; a task passes when every component is 1.
Component names mirror what the suite measures: tool selection, argument
accuracy, loop detection and the other terminal states (state and code),
abstention (state, citations), side effects (proposals), and efficiency.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from app.core.agentic.evalkit.suite import Expect


class Facts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: str
    code: str | None
    attempted_tools: list[str]
    calls: list[dict[str, Any]]  # {"tool": name, "arguments": {...}}
    proposals: int
    cited: list[str]
    steps: int


class Scores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: float
    code: float
    tool_selection: float
    argument_accuracy: float
    forbidden: float
    effects: float
    citations: float
    efficiency: float

    @property
    def passed(self) -> bool:
        return all(v == 1.0 for v in self.model_dump().values())

    @property
    def mean(self) -> float:
        values = list(self.model_dump().values())
        return round(sum(values) / len(values), 4)


def _tool_selection(attempted: list[str], expected: tuple[str, ...]) -> float:
    """1.0 for the exact sequence; otherwise the overlap of the two tool sets,
    halved, so a right set in the wrong order or count scores 0.5 at most."""
    if attempted == list(expected):
        return 1.0
    a, e = set(attempted), set(expected)
    if not a and not e:
        return 1.0
    return round(0.5 * len(a & e) / len(a | e), 4)


def _argument_accuracy(calls: list[dict[str, Any]], expected: tuple[dict[str, Any], ...]) -> float:
    if not expected:
        return 1.0
    hit = 0
    for want in expected:
        subset = want.get("subset", {})
        if any(
            call["tool"] == want["tool"]
            and all(call["arguments"].get(k) == v for k, v in subset.items())
            for call in calls
        ):
            hit += 1
    return round(hit / len(expected), 4)


def score(facts: Facts, expect: Expect) -> Scores:
    return Scores(
        state=1.0 if facts.state == expect.state else 0.0,
        code=1.0 if facts.code == expect.code else 0.0,
        tool_selection=_tool_selection(facts.attempted_tools, expect.tools),
        argument_accuracy=_argument_accuracy(facts.calls, expect.args),
        forbidden=0.0 if set(facts.attempted_tools) & set(expect.forbidden_tools) else 1.0,
        effects=1.0 if facts.proposals == expect.proposals else 0.0,
        citations=1.0 if set(facts.cited) == set(expect.cited) else 0.0,
        efficiency=1.0 if facts.steps <= expect.max_steps else 0.0,
    )
