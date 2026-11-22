"""Models the harness can drive: the reference trajectory, and three bad ones.

The reference policy plays each task's scripted decisions by step; it is what a
good model would do, and every task must pass with it (that checks the tasks
and the runtime together). The three baselines are negative controls: if the
scoring cannot tell them from the reference, it measures nothing. A real model
is another factory returning an AgentModel (see harness.run_suite).
"""

from __future__ import annotations

from app.core.agentic.evalkit.suite import EvalTask
from app.core.agentic.evalkit.trace import StepIndexedModel
from app.core.agentic.model import AgentModel, Decision, DecisionRequest


def reference(task: EvalTask) -> AgentModel:
    return StepIndexedModel([Decision(**item) for item in task.script])


class _Naive:
    """Answers at once, from nothing, with a confident score."""

    async def decide(self, request: DecisionRequest) -> Decision:
        return Decision(kind="final", answer="Easy. You scored 95% on this topic.", tokens=10)


class _Looping:
    """Searches the same thing for ever."""

    async def decide(self, request: DecisionRequest) -> Decision:
        return Decision(
            kind="tool",
            tool="search_course_material",
            arguments={"query": request.goal[:60]},
            tokens=10,
        )


class _Overreach:
    """Reaches for a tool it was not given, then cites a source it never saw."""

    async def decide(self, request: DecisionRequest) -> Decision:
        if request.step == 1:
            return Decision(
                kind="tool", tool="update_learning_progress", arguments={"change": {}}, tokens=10
            )
        return Decision(kind="final", answer="Done.", citations=("invented-chunk",), tokens=10)


def naive(task: EvalTask) -> AgentModel:
    return _Naive()


def looping(task: EvalTask) -> AgentModel:
    return _Looping()


def overreach(task: EvalTask) -> AgentModel:
    return _Overreach()


POLICIES = {"reference": reference, "naive": naive, "looping": looping, "overreach": overreach}
