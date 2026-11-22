"""Test support for the agent runtime: a scripted model, a task builder, and a
registry with the study tools over a small synthetic corpus."""

import asyncio
from collections.abc import Callable
from typing import Any

from app.core.agentic.contracts import (
    AgentBudget,
    AgentTask,
    CallerContext,
    CompletionCriteria,
    EvidenceRequirements,
    ExecutionPolicy,
    PlanItem,
    PracticeEvidence,
    SuppliedData,
    TopicRef,
)
from app.core.agentic.memory_ports import CorpusPassage, InMemoryBackendPort, InMemoryEvidencePort
from app.core.agentic.model import Decision, DecisionRequest
from app.core.agentic.ports import Ports
from app.core.agentic.registry import ToolRegistry
from app.core.agentic.tools import register_study_tools
from app.services.agentic.ports import ModelCallerAdapter
from tests.fake_models import OfflineModel

CORPUS = [
    CorpusPassage(
        "t1",
        "c-heat-1",
        "doc-thermo",
        "Heat transfer happens by conduction, convection and radiation.",
        3,
    ),
    CorpusPassage(
        "t1",
        "c-heat-2",
        "doc-thermo",
        "Conduction moves heat through a solid without moving the solid.",
        4,
    ),
    CorpusPassage(
        "t1",
        "c-lens-1",
        "doc-optics",
        "A converging lens focuses parallel rays at a focal point.",
        9,
    ),
]
PERMISSIONS = (
    "content:read",
    "assessment:generate",
    "progress:read",
    "progress:propose",
    "plan:read",
    "plan:propose",
)
ALL_TOOLS = (
    "search_course_material",
    "explain_concept",
    "generate_practice",
    "evaluate_response",
    "identify_knowledge_gaps",
    "create_study_plan",
    "recommend_next_activity",
    "update_learning_progress",
)


def tool(name: str, **arguments: Any) -> Decision:
    return Decision(kind="tool", tool=name, arguments=arguments, tokens=10)


def final(answer: str, *citations: str, tokens: int = 10) -> Decision:
    return Decision(kind="final", answer=answer, citations=citations, tokens=tokens)


class ScriptedModel:
    """Plays a script of decisions, or exceptions to raise, or callables given
    the request. Records every request it was asked."""

    def __init__(self, *script: Decision | BaseException | Callable[[DecisionRequest], Any]):
        self.script = list(script)
        self.requests: list[DecisionRequest] = []

    async def decide(self, request: DecisionRequest) -> Decision:
        self.requests.append(request)
        if not self.script:
            raise RuntimeError("script exhausted")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            result = item(request)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, BaseException):
                raise result
            return result  # type: ignore[no-any-return]
        return item


def context(**over: Any) -> CallerContext:
    base: dict[str, Any] = dict(
        tenant_id="t1",
        user_id="u1",
        permissions=PERMISSIONS,
        capabilities=("agentic_workflows",),
        denied_assets=(),
    )
    base.update(over)
    return CallerContext(**base)


def task(**over: Any) -> AgentTask:
    base: dict[str, Any] = dict(
        task_id="task1",
        execution_id="exec1",
        task_type="grounded_explanation",
        goal="Explain how heat moves through a solid",
        context=context(),
        required_capabilities=("agentic_workflows",),
        allowed_tools=("search_course_material", "identify_knowledge_gaps", "create_study_plan"),
    )
    base.update(over)
    return AgentTask(**base)


def registry(backend: InMemoryBackendPort | None = None, corpus=None) -> ToolRegistry:
    ports = Ports(
        retrieval=InMemoryEvidencePort(corpus or CORPUS),
        backend=backend or InMemoryBackendPort(),
        model=ModelCallerAdapter(OfflineModel(), "chat", None),
    )
    return register_study_tools(ToolRegistry(ports))


def supplied(**over: Any) -> SuppliedData:
    rows = tuple(
        PracticeEvidence(topic_id="heat", correct=ok, answered_at=i * 86_400.0, attempt_ref=f"a{i}")
        for i, ok in enumerate([False, False, False])
    )
    base: dict[str, Any] = dict(
        topics=(TopicRef(topic_id="heat", title="Heat transfer"),),
        practice_evidence=rows,
        plan=(
            PlanItem(
                item_id="p1", topic_id="heat", due_epoch_day=100, activity="practice", minutes=30
            ),
        ),
        today_epoch_day=100,
    )
    base.update(over)
    return SuppliedData(**base)


__all__ = [
    "ALL_TOOLS",
    "AgentBudget",
    "CompletionCriteria",
    "EvidenceRequirements",
    "ExecutionPolicy",
    "ScriptedModel",
    "context",
    "final",
    "registry",
    "supplied",
    "task",
    "tool",
]
