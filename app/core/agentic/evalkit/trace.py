"""Trace recording and replay.

A trace is the decisions a model made and what the tools returned, step by step.
Replaying one needs no model: ``ReplayAgentModel`` hands back the recorded
decisions by step, so a run (even one first made with a real model) is
reproduced exactly and can be re-scored. A trace of a real run holds the
model's answers and the tool arguments: treat it as student data. The suite's
own traces are synthetic.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from app.core.agentic.model import Decision, DecisionRequest, ModelUnavailable


class Trace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    suite_sha256: str
    events: list[dict[str, Any]] = []

    def decisions(self) -> list[Decision]:
        return [
            Decision.model_validate(e["decision"]) for e in self.events if e["type"] == "decision"
        ]


class TraceRecorder:
    def __init__(self, task_id: str, suite_sha256: str) -> None:
        self.trace = Trace(task_id=task_id, suite_sha256=suite_sha256)

    def __call__(self, event: dict[str, Any]) -> None:
        self.trace.events.append(event)


class StepIndexedModel:
    """Decisions by step number, not by call count: a run resumed after an
    interruption asks for step N+1 and gets it, whatever was asked before."""

    def __init__(self, decisions: list[Decision]) -> None:
        self._decisions = decisions

    async def decide(self, request: DecisionRequest) -> Decision:
        if request.step < 1 or request.step > len(self._decisions):
            raise ModelUnavailable()
        return self._decisions[request.step - 1]


class ReplayAgentModel(StepIndexedModel):
    def __init__(self, trace: Trace) -> None:
        super().__init__(trace.decisions())
