"""Run a suite against a model, score it, record traces.

``run_suite(suite, factory)`` is the whole interface: the factory takes an
``EvalTask`` and returns an ``AgentModel``. The scripted reference, the three
baselines and ``ReplayAgentModel`` are factories; so is a real model, built the
way the server builds one:

    def real(task):
        invocation = ...  # a ModelInvocationContext for this task
        return ModelCallerAgentModel(gateway_llm, "chat", invocation)

That factory belongs in a reviewed script that has provider approval, a budget
and a network allowance; this package has none of those and makes no call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.core.agentic.contracts import (
    AgentBudget,
    AgentTask,
    CallerContext,
    CallKind,
    CompletionCriteria,
    EvidenceRequirements,
    ExecutionPolicy,
    digest,
)
from app.core.agentic.evalkit.scoring import Facts, Scores, score
from app.core.agentic.evalkit.suite import TENANT, USER, EvalTask, Suite
from app.core.agentic.evalkit.trace import Trace, TraceRecorder
from app.core.agentic.memory_ports import (
    DeterministicModelPort,
    InMemoryBackendPort,
    InMemoryEvidencePort,
)
from app.core.agentic.model import AgentModel, Decision, DecisionRequest
from app.core.agentic.ports import Ports
from app.core.agentic.registry import RunControl, ToolRegistry
from app.core.agentic.runtime import AgentRunResult, AgentRuntime
from app.core.agentic.stores import InMemoryCheckpointStore, InMemoryResultStore
from app.core.agentic.tools import register_study_tools

ModelFactory = Callable[[EvalTask], AgentModel]


class TaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    category: str
    scores: Scores
    passed: bool
    mean: float
    facts: Facts
    result_digest: str


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite_name: str
    suite_version: int
    suite_sha256: str
    policy: str
    results: list[TaskResult]
    by_category: dict[str, float]
    pass_rate: float
    mean: float

    def digest(self) -> str:
        return digest(self.model_dump(mode="json"))


class _Cancelling:
    """Cancels the run when the model is asked for a given step."""

    def __init__(self, inner: AgentModel, control: RunControl, step: int) -> None:
        self._inner, self._control, self._step = inner, control, step

    async def decide(self, request: DecisionRequest) -> Decision:
        decision = await self._inner.decide(request)
        if request.step == self._step:
            self._control.cancel("user")
        return decision


class _CrashAfter(InMemoryCheckpointStore):
    """Saves until `limit` saves, then the process 'dies'."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self._limit = limit

    async def save(self, checkpoint: Any) -> None:
        if self.saves >= self._limit:
            raise OSError("process died")
        await super().save(checkpoint)


def _agent_task(suite: Suite, task: EvalTask) -> AgentTask:
    return AgentTask(
        task_id=task.id,
        execution_id=f"eval-{task.id}",
        task_type=task.task_type,
        goal=task.goal,
        context=CallerContext(
            tenant_id=TENANT,
            user_id=USER,
            permissions=task.permissions,
            capabilities=("agentic_workflows",),
            denied_assets=task.denied_assets,
            official_attempt_active=task.official_attempt_active,
        ),
        required_capabilities=("agentic_workflows",),
        allowed_tools=task.allowed_tools,
        evidence=EvidenceRequirements(**task.evidence),
        budget=AgentBudget(**task.budget),
        execution_policy=ExecutionPolicy(**task.policy),
        completion=CompletionCriteria(**task.completion),
        supplied=suite.supplied_for(task),
    )


def _registry(suite: Suite, backend: InMemoryBackendPort) -> ToolRegistry:
    ports = Ports(
        retrieval=InMemoryEvidencePort(list(suite.corpus)),
        backend=backend,
        model=DeterministicModelPort(),
    )
    return register_study_tools(ToolRegistry(ports))


async def run_task(
    suite: Suite, task: EvalTask, factory: ModelFactory
) -> tuple[TaskResult, Trace, AgentRunResult]:
    agent_task = _agent_task(suite, task)
    backend = InMemoryBackendPort()
    recorder = TraceRecorder(task.id, suite.sha256)
    control = RunControl()
    model = factory(task)
    if task.inject.cancel_at_step is not None:
        model = _Cancelling(model, control, task.inject.cancel_at_step)
    results = InMemoryResultStore()

    if task.inject.crash_after_saves is None:
        runtime = AgentRuntime(
            _registry(suite, backend),
            model,
            InMemoryCheckpointStore(),
            results,
            observer=recorder,
        )
        result = await runtime.run(agent_task, control)
    else:
        crashing = _CrashAfter(task.inject.crash_after_saves)
        first = AgentRuntime(_registry(suite, backend), model, crashing, results, observer=recorder)
        await first.run(agent_task, control)
        # A restart: fresh process state (registry, idempotency store), the
        # same backend, results and whatever checkpoints survived.
        survivor = InMemoryCheckpointStore()
        survivor._items = dict(crashing._items)
        second = AgentRuntime(
            _registry(suite, backend), factory(task), survivor, results, observer=recorder
        )
        result = await second.resume(agent_task, RunControl())

    tool_calls = [c for c in result.checkpoint.calls if c.kind is CallKind.TOOL]
    asked = [e for e in recorder.trace.events if e["type"] == "tool"]
    facts = Facts(
        state=result.state.value,
        code=result.outcome_code,
        attempted_tools=[c.name for c in tool_calls],
        calls=[{"tool": e["tool"], "arguments": e["arguments"]} for e in asked],
        proposals=len(backend.proposals),
        cited=[c["chunk_id"] for c in result.citations],
        steps=result.budget.steps,
    )
    scores = score(facts, task.expect)
    outcome = TaskResult(
        task_id=task.id,
        category=task.category,
        scores=scores,
        passed=scores.passed,
        mean=scores.mean,
        facts=facts,
        result_digest=digest(
            {
                "state": facts.state,
                "code": facts.code,
                "output": result.output,
                "cited": facts.cited,
                "budget": result.budget.model_dump(),
            }
        ),
    )
    return outcome, recorder.trace, result


async def run_suite(
    suite: Suite, factory: ModelFactory, policy: str = "custom"
) -> tuple[Report, dict[str, Trace]]:
    results: list[TaskResult] = []
    traces: dict[str, Trace] = {}
    for task in suite.tasks:
        outcome, trace, _ = await run_task(suite, task, factory)
        results.append(outcome)
        traces[task.id] = trace
    by_category: dict[str, list[float]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r.mean)
    report = Report(
        suite_name=suite.name,
        suite_version=suite.version,
        suite_sha256=suite.sha256,
        policy=policy,
        results=results,
        by_category={k: round(sum(v) / len(v), 4) for k, v in sorted(by_category.items())},
        pass_rate=round(sum(r.passed for r in results) / len(results), 4),
        mean=round(sum(r.mean for r in results) / len(results), 4),
    )
    return report, traces
