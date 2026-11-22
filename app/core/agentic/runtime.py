"""The agent runtime: a bounded, checkpointed, cancellable loop.

One run, from an AgentTask to an AgentRunResult:

    refuse? -> RUNNING -> [ decide -> (tool | final) ]* -> a terminal state

Every state change is a validated ExecutionCheckpoint, saved through the
CheckpointStore before the run goes on, so a run that is interrupted can be
resumed (``resume``) from where it was, with its budget already spent, never
reset. What a run guarantees:

- Bounded: steps (model decisions), iterations (tool rounds), tokens, cost and
  retries each have a ceiling in the task; the first to be reached ends the run
  in BUDGET_EXHAUSTED, with whatever tool results it gathered marked partial.
- One deadline and one cancellation, from the RunControl, passed to every model
  call and every tool call. Cancelling mid-call stops it; a call stopped in
  flight is recorded as such, and its idempotency key is what settles it.
- Only the task's allowed tools. A tool outside the list is refused by the
  registry, and counts as a wasted step.
- Loops end: the same call (tool and arguments) N times, or N tool rounds in a
  row that return nothing new, end the run in LOOP_DETECTED.
- No duplicated side effects: the registry replays a keyed call's first outcome,
  retries reuse the key, and a resumed run re-issues an interrupted call with
  the key it had.
- Sanitised: results carry a code and a fixed message. No exception's text, no
  provider text, no argument and no passage reaches a result or a log.
- The foundation checks (checks.py) decide whether a final answer stands: never
  an invented source, grade or mastery score; abstain when the evidence is not
  enough.
- Exam mode: while an official attempt is active, a run is refused before any
  model or tool call.

Nothing here knows about a route, a database or a provider. It is portable to a
device in logic; where it is server-shaped (asyncio, the wall-clock deadline) is
recorded in docs/agentic-architecture.md.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from app.core.agentic.checks import Ledger, check_final, numbers_in
from app.core.agentic.contracts import (
    TERMINAL_STATES,
    AgentTask,
    BudgetConsumed,
    CallKind,
    CallRecord,
    CallStatus,
    CancellationState,
    ExecutionCheckpoint,
    ExecutionState,
    ResultRef,
    can_transition,
    digest,
)
from app.core.agentic.grounding import NOT_COVERED
from app.core.agentic.model import (
    AgentModel,
    Decision,
    DecisionRequest,
    MalformedDecision,
    Turn,
    observation_text,
)
from app.core.agentic.registry import RunControl, ToolOutcome, ToolRegistry
from app.core.agentic.stores import (
    CheckpointStore,
    InMemoryCheckpointStore,
    InMemoryResultStore,
    ResultStore,
)
from app.core.safe_log import describe

logger = logging.getLogger(__name__)

# What a result says about how a run ended: one fixed message per code.
MESSAGES: dict[str, str] = {
    "EXAM_MODE_ACTIVE": "Study assistance is not available during an official attempt.",
    "CAPABILITY_DENIED": "This kind of study help is not available on your plan.",
    "PERMISSION_DENIED": "You do not have access to everything this task needs.",
    "TOOL_NOT_FOUND": "This task needs a tool that is not available.",
    "EXECUTION_EXISTS": "This run already exists; resume it instead.",
    "EXECUTION_IN_PROGRESS": "This run is already in progress.",
    "RESUME_NOT_POSSIBLE": "There is no run to resume.",
    "RESUME_MISMATCH": "This run does not belong to this task.",
    "RESUME_STATE_LOST": "The run's saved state is incomplete, so it cannot resume.",
    "RESULT_TAMPERED": "The run's saved results do not match, so it cannot resume.",
    "CANCELLED": "The run was cancelled.",
    "DEADLINE_EXPIRED": "The run ran out of time.",
    "STEP_BUDGET": "The run used its step budget before finishing.",
    "ITERATION_BUDGET": "The run used its tool budget before finishing.",
    "TOKEN_BUDGET": "The run used its token budget before finishing.",
    "COST_BUDGET": "The run used its cost budget before finishing.",
    "RETRY_BUDGET": "The run used its retry budget before finishing.",
    "DOOM_LOOP": "The run repeated the same step and was stopped.",
    "NO_PROGRESS": "The run stopped making progress and was stopped.",
    "MODEL_UNAVAILABLE": "The assistant could not be reached.",
    "MODEL_OUTPUT_INVALID": "The assistant did not give a usable answer.",
    "INCOMPLETE": "The assistant finished without completing the required steps.",
    "INVENTED_SOURCE": "The answer cited a source it was not given, so it was withheld.",
    "UNSUPPORTED_FIGURE": "The answer stated a score or grade that was not computed, so it was withheld.",
    "INSUFFICIENT_EVIDENCE": NOT_COVERED,
    "STORE_UNAVAILABLE": "The run's state could not be saved, so it was stopped.",
}
_CANCEL_STOP = "CANCELLED"
_DENIED_CODES = frozenset(
    {
        "TOOL_NOT_FOUND",
        "TOOL_NOT_ALLOWED",
        "PERMISSION_DENIED",
        "CAPABILITY_DENIED",
        "EXAM_MODE_ACTIVE",
        "SIDE_EFFECT_DISALLOWED",
        "TENANT_MISMATCH",
        "IDEMPOTENCY_CONFLICT",
        "INVENTED_SOURCE",
    }
)
Observer = Callable[[dict[str, Any]], None]


class AgentRunResult(BaseModel):
    """What a run returns. The checkpoint is the full, auditable record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_id: str
    state: ExecutionState
    output: str | None = None
    citations: tuple[dict[str, Any], ...] = ()
    # Budget or loop ended the run, and these tool results were gathered first.
    partial: bool = False
    partial_refs: tuple[str, ...] = ()
    outcome_code: str | None = None
    message: str | None = None
    budget: BudgetConsumed
    checkpoint: ExecutionCheckpoint


class _Stop(Exception):
    def __init__(self, state: ExecutionState, code: str | None) -> None:
        super().__init__(code or "")
        self.state = state
        self.code = code


class _Run:
    def __init__(self, task: AgentTask, cp: ExecutionCheckpoint, control: RunControl) -> None:
        self.task = task
        self.cp = cp
        self.control = control
        self.ledger = Ledger()
        self.produced: set[float] = set()
        self.seen: set[str] = set()
        self.turns: list[Turn] = []
        self.no_progress = 0
        self.failures = 0
        self.premature = 0
        self.notes: list[str] = []


async def _bounded(
    work: Any, control: RunControl, timeout: float
) -> tuple[Literal["ok", "cancelled", "timeout"], Any]:
    """Await `work`, stopping at the timeout or when the run is cancelled."""
    task = asyncio.ensure_future(work)
    waiter = asyncio.ensure_future(control.wait_cancelled())
    try:
        done, _ = await asyncio.wait(
            {task, waiter}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
    except asyncio.CancelledError:
        task.cancel()
        waiter.cancel()
        raise
    waiter.cancel()
    if task in done:
        return "ok", task.result()  # raises the work's own exception
    task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task
    return ("cancelled" if control.cancelled else "timeout"), None


class AgentRuntime:
    def __init__(
        self,
        registry: ToolRegistry,
        model: AgentModel,
        checkpoints: CheckpointStore | None = None,
        results: ResultStore | None = None,
        observer: Observer | None = None,
    ) -> None:
        self.registry = registry
        self.model = model
        self.checkpoints: CheckpointStore = checkpoints or InMemoryCheckpointStore()
        self.results: ResultStore = results or InMemoryResultStore()
        self._observer = observer
        self._active: set[str] = set()

    # --- entry points -------------------------------------------------------

    async def run(self, task: AgentTask, control: RunControl | None = None) -> AgentRunResult:
        control = control or RunControl.from_epoch(task.deadline_epoch)
        execution_id = task.execution_id
        if execution_id in self._active:
            return self._refused_unsaved(task, "EXECUTION_IN_PROGRESS", None)
        self._active.add(execution_id)
        try:
            existing = await self._load(execution_id)
            if existing is not None:
                return self._refused_unsaved(task, "EXECUTION_EXISTS", existing)
            cp = self._new_checkpoint(task)
            refusal = self._refusal(task)
            if refusal:
                cp = self._with(cp, state=ExecutionState.REFUSED, outcome_code=refusal)
                await self._save_best_effort(cp)
                return self._result(task, cp, None, [], False)
            cp = self._with(cp, state=ExecutionState.RUNNING)
            run = _Run(task, cp, control)
            return await self._drive(run)
        finally:
            self._active.discard(execution_id)

    async def resume(self, task: AgentTask, control: RunControl | None = None) -> AgentRunResult:
        control = control or RunControl.from_epoch(task.deadline_epoch)
        execution_id = task.execution_id
        if execution_id in self._active:
            return self._refused_unsaved(task, "EXECUTION_IN_PROGRESS", None)
        self._active.add(execution_id)
        try:
            cp = await self._load(execution_id)
            if cp is None:
                return self._refused_unsaved(task, "RESUME_NOT_POSSIBLE", None)
            if (
                cp.task_digest != task.digest()
                or cp.tenant_id != task.context.tenant_id
                or cp.user_id != task.context.user_id
            ):
                return self._refused_unsaved(task, "RESUME_MISMATCH", cp)
            if cp.state in TERMINAL_STATES:
                return await self._finished_result(task, cp)
            if self._refusal(task):  # exam mode may have begun since
                return self._refused_unsaved(task, self._refusal(task) or "", cp)
            if cp.state is ExecutionState.PENDING:
                cp = self._with(cp, state=ExecutionState.RUNNING)
            run = _Run(task, cp, control)
            try:
                pending = await self._rebuild(run)
                if pending is not None:
                    await self._reissue(run, pending)
            except _Stop as stop:
                return await self._end(run, stop.state, stop.code)
            return await self._drive(run)
        finally:
            self._active.discard(execution_id)

    # --- the loop -----------------------------------------------------------

    async def _drive(self, run: _Run) -> AgentRunResult:
        try:
            await self._save(run)
            while True:
                self._check_stop(run)
                budget = run.task.budget
                used = run.cp.budget_consumed
                if used.steps >= budget.max_steps:
                    raise _Stop(ExecutionState.BUDGET_EXHAUSTED, "STEP_BUDGET")
                if used.tokens >= budget.max_tokens:
                    raise _Stop(ExecutionState.BUDGET_EXHAUSTED, "TOKEN_BUDGET")
                if budget.max_cost_usd is not None and used.cost_usd >= budget.max_cost_usd:
                    raise _Stop(ExecutionState.BUDGET_EXHAUSTED, "COST_BUDGET")
                decision = await self._decide(run)
                self._check_stop(run)  # a cancel during the decision wins over its answer
                if decision.kind == "tool":
                    if await self._tool_step(run, decision):
                        return await self._end(run, ExecutionState.COMPLETED, None)
                else:
                    done = await self._final_step(run, decision)
                    if done is not None:
                        return done
        except _Stop as stop:
            return await self._end(run, stop.state, stop.code)

    def _check_stop(self, run: _Run) -> None:
        if run.control.cancelled:
            if run.control.reason == "deadline":
                raise _Stop(ExecutionState.TIMED_OUT, "DEADLINE_EXPIRED")
            raise _Stop(ExecutionState.CANCELLED, _CANCEL_STOP)
        if run.control.expired():
            raise _Stop(ExecutionState.TIMED_OUT, "DEADLINE_EXPIRED")

    async def _decide(self, run: _Run) -> Decision:
        task, policy = run.task, run.task.execution_policy
        while True:
            self._check_stop(run)
            used = run.cp.budget_consumed
            request = DecisionRequest(
                goal=task.goal,
                task_type=task.task_type,
                tools=tuple(
                    c.describe_for_model()
                    for c in self.registry.available(task.context, task.allowed_tools)
                ),
                turns=tuple(run.turns),
                step=used.steps + 1,
                steps_left=task.budget.max_steps - used.steps,
                iterations_left=task.budget.max_iterations - used.iterations,
                evidence_ids=tuple(sorted(run.ledger.passages)),
                notes=tuple(run.notes),
            )
            failure: str | None = None
            decision: Decision | None = None
            run_bound = run.control.binds(policy.step_timeout_seconds)
            try:
                status, value = await _bounded(
                    self.model.decide(request),
                    run.control,
                    run.control.tighten(policy.step_timeout_seconds),
                )
                if status == "cancelled":
                    self._check_stop(run)
                elif status == "timeout":
                    if run_bound or run.control.expired():
                        raise _Stop(ExecutionState.TIMED_OUT, "DEADLINE_EXPIRED")
                    failure = "MODEL_UNAVAILABLE"
                else:
                    decision = value
            except MalformedDecision:
                failure = "MODEL_OUTPUT_INVALID"
            except _Stop:
                raise
            except Exception as exc:  # any model failure: type and place only
                logger.warning("Agent model call failed: %s", describe(exc))
                failure = "MODEL_UNAVAILABLE"
            if not isinstance(decision, Decision) and failure is None:
                failure = "MODEL_OUTPUT_INVALID"
            if failure is not None:
                run.failures += 1
                self._charge(run, retries=1)
                await self._record_model(run, CallStatus.FAILED, failure, 0)
                if run.cp.budget_consumed.retries > task.budget.max_retries:
                    raise _Stop(ExecutionState.BUDGET_EXHAUSTED, "RETRY_BUDGET")
                if run.failures >= policy.max_consecutive_failures:
                    raise _Stop(ExecutionState.FAILED, failure)
                continue
            assert decision is not None
            run.failures = 0
            self._charge(
                run, steps=1, model_calls=1, tokens=decision.tokens, cost_usd=decision.cost_usd
            )
            run.cp = self._with(run.cp, step=run.cp.step + 1)
            await self._record_model(run, CallStatus.OK, None, decision.tokens)
            self._emit(
                {
                    "type": "decision",
                    "step": run.cp.step,
                    "decision": decision.model_dump(mode="json"),
                }
            )
            return decision

    # --- tools --------------------------------------------------------------

    async def _tool_step(self, run: _Run, decision: Decision) -> bool:
        """Run the tool the model chose. True if the task is complete without a
        final answer (a task that does not require one)."""
        task, policy = run.task, run.task.execution_policy
        name = str(decision.tool)
        args = decision.arguments
        used = run.cp.budget_consumed
        if used.iterations >= task.budget.max_iterations:
            raise _Stop(ExecutionState.BUDGET_EXHAUSTED, "ITERATION_BUDGET")
        try:
            args_digest = digest(args)
        except (TypeError, ValueError):
            args_digest = digest("unserialisable")
            args = {}
            invalid = True
        else:
            invalid = False
        same = sum(
            1
            for c in run.cp.calls
            if c.kind is CallKind.TOOL and c.name == name and c.args_digest == args_digest
        )
        if same + 1 >= policy.doom_loop_repeat_limit:
            raise _Stop(ExecutionState.LOOP_DETECTED, "DOOM_LOOP")

        call_id = f"{task.execution_id}-t{used.tool_calls + 1}"
        self._charge(run, iterations=1, tool_calls=1)
        invented = self._invented_refs(run, args)
        if invalid or invented:
            code = "INVALID_ARGUMENTS" if invalid else "INVENTED_SOURCE"
            outcome = ToolOutcome(
                call_id=call_id,
                tool=name[:64],
                ok=False,
                error_code=code,
                message=(
                    "The tool's arguments were not valid."
                    if invalid
                    else "A source reference was not one that retrieval returned."
                ),
            )
            record = CallRecord(
                call_id=call_id,
                step=run.cp.step,
                kind=CallKind.TOOL,
                name=name[:64],
                args_digest=args_digest,
                status=CallStatus.DENIED,
                error_code=code,
            )
            run.cp = self._with(run.cp, calls=(*run.cp.calls, record))
            await self._absorb_outcome(run, record, name, args, outcome)
            return self._tools_complete(run)

        await self.results.put(f"intent-{call_id}", {"tool": name, "args": args})
        record = CallRecord(
            call_id=call_id,
            step=run.cp.step,
            kind=CallKind.TOOL,
            name=name[:64],
            args_digest=args_digest,
            status=CallStatus.STARTED,
        )
        run.cp = self._with(run.cp, calls=(*run.cp.calls, record))
        await self._save(run)
        return await self._issue(run, record, name, args)

    async def _issue(self, run: _Run, record: CallRecord, name: str, args: dict[str, Any]) -> bool:
        task = run.task
        outcome = await self.registry.execute(
            name,
            args,
            task.context,
            call_id=record.call_id,
            execution_id=task.execution_id,
            control=run.control,
            allowed=task.allowed_tools,
            supplied=task.supplied,
        )
        stopped = outcome.error_code in ("EXECUTION_CANCELLED", "DEADLINE_EXPIRED") or (
            not outcome.effect_known and run.control.expired()
        )
        if stopped:
            # Stopped in flight: the effect, if any, is unknown. Record it and
            # end; the same idempotency key is what settles it on a resume.
            self._replace(run, record, status=CallStatus.CANCELLED, error_code=outcome.error_code)
            self._charge(run, tokens=outcome.model_tokens, retries=outcome.retries)
            await self._save(run)
            self._check_stop(run)
            if outcome.error_code == "EXECUTION_CANCELLED":
                raise _Stop(ExecutionState.CANCELLED, _CANCEL_STOP)
            raise _Stop(ExecutionState.TIMED_OUT, "DEADLINE_EXPIRED")
        status = (
            CallStatus.REPLAYED
            if outcome.replayed
            else (
                CallStatus.OK
                if outcome.ok
                else CallStatus.DENIED if outcome.error_code in _DENIED_CODES else CallStatus.FAILED
            )
        )
        self._replace(
            run,
            record,
            status=status,
            error_code=outcome.error_code,
            retries=outcome.retries,
            tokens=outcome.model_tokens,
        )
        record = next(c for c in run.cp.calls if c.call_id == record.call_id)
        self._charge(run, tokens=outcome.model_tokens, retries=outcome.retries)
        await self._absorb_outcome(run, record, name, args, outcome)
        if run.cp.budget_consumed.retries > task.budget.max_retries:
            raise _Stop(ExecutionState.BUDGET_EXHAUSTED, "RETRY_BUDGET")
        return self._tools_complete(run)

    async def _absorb_outcome(
        self, run: _Run, record: CallRecord, name: str, args: dict[str, Any], outcome: ToolOutcome
    ) -> None:
        payload: dict[str, Any] = {
            "tool": name,
            "args": args,
            "ok": outcome.ok,
            "data": outcome.data,
            "failure": None if outcome.ok else outcome.failure(),
        }
        ref_id = f"res-{record.call_id}"
        sha = await self.results.put(ref_id, payload)
        run.cp = self._with(
            run.cp,
            results=(
                *run.cp.results,
                ResultRef(ref_id=ref_id, call_id=record.call_id, step=record.step, sha256=sha),
            ),
        )
        self._replace(run, record, result_ref=ref_id)
        self._absorb(run, record, payload)
        await self._save(run)
        self._emit(
            {
                "type": "tool",
                "step": record.step,
                "call_id": record.call_id,
                "tool": name,
                "arguments": args,
                "ok": outcome.ok,
                "error_code": outcome.error_code,
                "replayed": outcome.replayed,
                "result_digest": digest(outcome.data) if outcome.ok else None,
            }
        )
        if run.no_progress >= run.task.execution_policy.no_progress_limit:
            raise _Stop(ExecutionState.LOOP_DETECTED, "NO_PROGRESS")

    def _absorb(self, run: _Run, record: CallRecord, payload: dict[str, Any]) -> None:
        """Fold one finished call into the run: ledger, figures, progress, history.
        Live calls and a resume's rebuild both come through here."""
        contract = self.registry.contract(record.name)
        data = payload.get("data")
        progress = False
        if (
            payload.get("ok")
            and isinstance(data, dict)
            and record.status is not CallStatus.REPLAYED
        ):
            new_passages = run.ledger.add_from_output(contract, data) if contract else 0
            run.produced |= numbers_in(data)
            fingerprint = digest(data)
            progress = new_passages > 0 or fingerprint not in run.seen
            run.seen.add(fingerprint)
        run.no_progress = 0 if progress else run.no_progress + 1
        observed = (
            {"ok": True, "data": data}
            if payload.get("ok")
            else payload.get("failure") or {"ok": False}
        )
        run.turns.append(
            Turn(
                step=record.step,
                tool=record.name,
                arguments=payload.get("args") or {},
                observation=observation_text(observed),
            )
        )

    def _tools_complete(self, run: _Run) -> bool:
        completion = run.task.completion
        if completion.require_final_answer or not completion.required_tools:
            return False
        return set(completion.required_tools) <= self._succeeded(run)

    @staticmethod
    def _succeeded(run: _Run) -> set[str]:
        return {
            c.name
            for c in run.cp.calls
            if c.kind is CallKind.TOOL and c.status in (CallStatus.OK, CallStatus.REPLAYED)
        }

    @staticmethod
    def _invented_refs(run: _Run, args: Any) -> bool:
        """A `source_refs` argument naming a chunk retrieval did not return in
        this run: the model must not invent a source, even as an argument."""
        if isinstance(args, dict):
            for key, value in args.items():
                if key == "source_refs":
                    if not isinstance(value, list) or any(
                        not isinstance(v, str) or not run.ledger.has(v) for v in value
                    ):
                        return True
                elif AgentRuntime._invented_refs(run, value):
                    return True
        elif isinstance(args, list):
            return any(AgentRuntime._invented_refs(run, item) for item in args)
        return False

    # --- the end of a run -----------------------------------------------------

    async def _final_step(self, run: _Run, decision: Decision) -> AgentRunResult | None:
        task, policy = run.task, run.task.execution_policy
        missing = [t for t in task.completion.required_tools if t not in self._succeeded(run)]
        if missing:
            run.premature += 1
            run.notes = ["A required step has not been done yet. Do it before finishing."]
            if run.premature >= policy.max_consecutive_failures:
                raise _Stop(ExecutionState.FAILED, "INCOMPLETE")
            return None
        verdict = check_final(
            str(decision.answer),
            list(decision.citations),
            run.ledger,
            run.produced,
            task.evidence,
        )
        if verdict.verdict == "violation":
            raise _Stop(ExecutionState.FAILED, verdict.code)
        payload = {
            "answer": verdict.answer,
            "citations": [run.ledger.passages[c] for c in verdict.cited],
        }
        ref_id = f"res-{task.execution_id}-final"
        sha = await self.results.put(ref_id, payload)
        final_id = f"{task.execution_id}-final"
        record = CallRecord(
            call_id=final_id,
            step=run.cp.step,
            kind=CallKind.MODEL,
            name="final",
            args_digest=digest("final"),
            status=CallStatus.OK,
            result_ref=ref_id,
        )
        run.cp = self._with(
            run.cp,
            calls=(*run.cp.calls, record),
            results=(
                *run.cp.results,
                ResultRef(ref_id=ref_id, call_id=final_id, step=record.step, sha256=sha),
            ),
        )
        state = ExecutionState.COMPLETED if verdict.verdict == "ok" else ExecutionState.ABSTAINED
        return await self._end(run, state, verdict.code, payload=payload)

    async def _end(
        self,
        run: _Run,
        state: ExecutionState,
        code: str | None,
        payload: dict[str, Any] | None = None,
    ) -> AgentRunResult:
        control = run.control
        changes: dict[str, Any] = {"state": state, "outcome_code": code}
        if state is ExecutionState.CANCELLED:
            changes["cancellation"] = CancellationState(
                requested=True, reason=control.reason or "user", at_step=run.cp.step
            )
        cp = self._with(run.cp, **changes)
        run.cp = cp
        await self._save_best_effort(cp)
        partial_refs: list[str] = []
        if (
            state
            in (
                ExecutionState.BUDGET_EXHAUSTED,
                ExecutionState.LOOP_DETECTED,
                ExecutionState.TIMED_OUT,
            )
            and run.task.execution_policy.allow_partial_result
        ):
            partial_refs = [
                c.result_ref
                for c in cp.calls
                if c.kind is CallKind.TOOL and c.status is CallStatus.OK and c.result_ref
            ]
        self._emit({"type": "end", "state": state.value, "code": code})
        return self._result(run.task, cp, payload, partial_refs, bool(partial_refs))

    def _result(
        self,
        task: AgentTask,
        cp: ExecutionCheckpoint,
        payload: dict[str, Any] | None,
        partial_refs: list[str],
        partial: bool,
    ) -> AgentRunResult:
        code = cp.outcome_code
        return AgentRunResult(
            execution_id=cp.execution_id,
            state=cp.state,
            output=payload.get("answer") if payload else None,
            citations=tuple(payload.get("citations", [])) if payload else (),
            partial=partial,
            partial_refs=tuple(partial_refs),
            outcome_code=code,
            message=MESSAGES.get(code, "The run did not complete.") if code else None,
            budget=cp.budget_consumed,
            checkpoint=cp,
        )

    def _refused_unsaved(
        self, task: AgentTask, code: str, existing: ExecutionCheckpoint | None
    ) -> AgentRunResult:
        """A refusal that must not touch a stored checkpoint (another run owns it)."""
        shown = existing or self._with(
            self._new_checkpoint(task), state=ExecutionState.REFUSED, outcome_code=code
        )
        return AgentRunResult(
            execution_id=task.execution_id,
            state=ExecutionState.REFUSED,
            outcome_code=code,
            message=MESSAGES.get(code, "The run was refused."),
            budget=shown.budget_consumed,
            checkpoint=shown,
        )

    async def _finished_result(self, task: AgentTask, cp: ExecutionCheckpoint) -> AgentRunResult:
        """A run that already ended: its result again, with nothing re-executed."""
        payload = None
        for call in cp.calls:
            if call.name == "final" and call.result_ref:
                payload = await self.results.get(call.result_ref)
        partial_refs = [
            c.result_ref
            for c in cp.calls
            if c.kind is CallKind.TOOL and c.status is CallStatus.OK and c.result_ref
        ]
        partial = (
            cp.state
            in (
                ExecutionState.BUDGET_EXHAUSTED,
                ExecutionState.LOOP_DETECTED,
                ExecutionState.TIMED_OUT,
            )
            and task.execution_policy.allow_partial_result
            and bool(partial_refs)
        )
        return self._result(task, cp, payload, partial_refs if partial else [], partial)

    # --- resume -----------------------------------------------------------------

    async def _rebuild(self, run: _Run) -> CallRecord | None:
        """Re-derive the run's working state from its checkpoint and stored
        results. Returns the interrupted tool call, if the run stopped mid-call."""
        refs = {r.ref_id: r for r in run.cp.results}
        pending: CallRecord | None = None
        for record in run.cp.calls:
            if record.kind is not CallKind.TOOL:
                continue
            if record.result_ref:
                payload = await self.results.get(record.result_ref)
                ref = refs.get(record.result_ref)
                if payload is None or ref is None:
                    raise _Stop(ExecutionState.FAILED, "RESUME_STATE_LOST")
                if digest(payload) != ref.sha256:
                    raise _Stop(ExecutionState.FAILED, "RESULT_TAMPERED")
                self._absorb(run, record, payload)
            elif record.status is CallStatus.STARTED:
                pending = record
            else:
                raise _Stop(ExecutionState.FAILED, "RESUME_STATE_LOST")
        return pending

    async def _reissue(self, run: _Run, record: CallRecord) -> None:
        intent = await self.results.get(f"intent-{record.call_id}")
        if intent is None or intent.get("tool") != record.name:
            raise _Stop(ExecutionState.FAILED, "RESUME_STATE_LOST")
        if digest(intent.get("args", {})) != record.args_digest:
            raise _Stop(ExecutionState.FAILED, "RESULT_TAMPERED")
        # The same call, with the same id: a keyed tool answers from its first
        # outcome or settles its unknown one under the same key.
        if await self._issue(run, record, record.name, intent.get("args", {})):
            raise _Stop(ExecutionState.COMPLETED, None)

    # --- bookkeeping --------------------------------------------------------------

    def _refusal(self, task: AgentTask) -> str | None:
        context = task.context
        if context.official_attempt_active:
            return "EXAM_MODE_ACTIVE"
        if not set(task.required_capabilities) <= set(context.capabilities):
            return "CAPABILITY_DENIED"
        if any(self.registry.contract(t) is None for t in task.allowed_tools):
            return "TOOL_NOT_FOUND"
        usable = {c.name for c in self.registry.available(context, task.allowed_tools)}
        if not set(task.completion.required_tools) <= usable:
            return "PERMISSION_DENIED"
        return None

    @staticmethod
    def _new_checkpoint(task: AgentTask) -> ExecutionCheckpoint:
        return ExecutionCheckpoint(
            execution_id=task.execution_id,
            task_id=task.task_id,
            task_digest=task.digest(),
            tenant_id=task.context.tenant_id,
            user_id=task.context.user_id,
        )

    @staticmethod
    def _with(cp: ExecutionCheckpoint, **changes: Any) -> ExecutionCheckpoint:
        """A new checkpoint, validated, with a legal state change only."""
        new_state = changes.get("state")
        if new_state is not None and new_state is not cp.state:
            if not can_transition(cp.state, new_state):
                raise ValueError("illegal state change")
        return ExecutionCheckpoint.model_validate({**cp.model_dump(), **changes})

    def _charge(self, run: _Run, **delta: float) -> None:
        used = run.cp.budget_consumed.model_dump()
        for key, amount in delta.items():
            used[key] = used[key] + amount
        run.cp = self._with(
            run.cp, budget_consumed=BudgetConsumed(**used), retry_count=int(used["retries"])
        )

    def _replace(self, run: _Run, record: CallRecord, **changes: Any) -> None:
        updated = record.model_copy(update=changes)
        run.cp = self._with(
            run.cp,
            calls=tuple(updated if c.call_id == record.call_id else c for c in run.cp.calls),
        )

    async def _record_model(
        self, run: _Run, status: CallStatus, code: str | None, tokens: int
    ) -> None:
        n = run.cp.budget_consumed.model_calls + run.cp.budget_consumed.retries
        record = CallRecord(
            call_id=f"{run.task.execution_id}-m{n}-{status.value}",
            step=run.cp.step,
            kind=CallKind.MODEL,
            name="decision",
            args_digest=digest({"step": run.cp.step, "turns": len(run.turns)}),
            status=status,
            error_code=code,
            tokens=tokens,
        )
        if all(c.call_id != record.call_id for c in run.cp.calls):
            run.cp = self._with(run.cp, calls=(*run.cp.calls, record))
        await self._save(run)

    async def _load(self, execution_id: str) -> ExecutionCheckpoint | None:
        try:
            return await self.checkpoints.load(execution_id)
        except Exception as exc:
            logger.error("Agent checkpoint load failed: %s", describe(exc))
            return None

    async def _save(self, run: _Run) -> None:
        try:
            await self.checkpoints.save(run.cp)
        except Exception as exc:
            logger.error("Agent checkpoint save failed: %s", describe(exc))
            raise _Stop(ExecutionState.FAILED, "STORE_UNAVAILABLE") from None

    async def _save_best_effort(self, cp: ExecutionCheckpoint) -> None:
        try:
            await self.checkpoints.save(cp)
        except Exception as exc:
            logger.error("Agent checkpoint save failed: %s", describe(exc))

    def _emit(self, event: dict[str, Any]) -> None:
        if self._observer is None:
            return
        try:
            self._observer(event)
        except Exception as exc:  # an observer never changes a run
            logger.warning("Agent observer failed: %s", describe(exc))
