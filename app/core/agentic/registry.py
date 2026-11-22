"""The typed tool registry: the one path every tool call takes.

register() accepts a tool only as a ToolContract plus a handler. execute()
then does, in this order and for every call, whoever makes it (the runtime, an
MCP client, a test):

1. look the tool up, and check the task's allow-list;
2. refuse during an official attempt (exam mode);
3. check the caller's permissions and capabilities, from the signed context;
4. refuse a side-effect class the registry was not configured to run;
5. check the run is not cancelled or past its deadline;
6. check the arguments against the input schema (and that none names another
   tenant or user);
7. for a keyed tool, find the idempotency key and return the first outcome for
   a repeat, without a second effect;
8. run the handler with the tighter of its own deadline and what remains of the
   run's, retrying only what the contract says is transient;
9. check the result against the output schema and size;
10. write one audit record, whichever way the call ended.

A handler gets a ToolCall: the caller's context, the checked arguments and a
ToolServices whose doors (retrieval, backend, model) open only if the contract
declared them. There is no database door. Errors reach the model as a code and a
fixed message; an exception's text goes nowhere (``describe`` only, as in #52).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.core.agentic.contracts import (
    CallerContext,
    Idempotency,
    RetryClass,
    SideEffect,
    SuppliedData,
    ToolContract,
    canonical_json,
    digest,
)
from app.core.agentic.ports import (
    BoundaryViolation,
    MeteredModel,
    Ports,
    ToolServices,
)
from app.core.agentic.schema import validate
from app.core.safe_log import describe

logger = logging.getLogger(__name__)

# What the model reads about a failure: a code, a retry flag, and fixed text.
MESSAGES: dict[str, str] = {
    "TOOL_NOT_FOUND": "The tool is not available.",
    "TOOL_NOT_ALLOWED": "The tool is not allowed for this task.",
    "PERMISSION_DENIED": "The tool is not permitted here.",
    "CAPABILITY_DENIED": "The tool is not permitted here.",
    "EXAM_MODE_ACTIVE": "Study tools are not available during an official attempt.",
    "SIDE_EFFECT_DISALLOWED": "The tool is not permitted here.",
    "EXECUTION_CANCELLED": "The run was cancelled.",
    "DEADLINE_EXPIRED": "The run ran out of time.",
    "INVALID_ARGUMENTS": "The tool's arguments were not valid.",
    "TENANT_MISMATCH": "The tool can act only within the caller's institution.",
    "IDEMPOTENCY_CONFLICT": "This request key was already used for a different request.",
    "MISSING_SCOPE": "The tool needs the caller's access scope, which this run does not have.",
    "TOOL_TIMEOUT": "The tool did not finish in time.",
    "TOOL_UNAVAILABLE": "The tool's service is not available right now.",
    "TOOL_FAILED": "The tool could not complete this operation.",
    "TOOL_OUTPUT_INVALID": "The tool returned something that could not be used.",
}
TRANSIENT_CODES = frozenset({"TOOL_TIMEOUT", "TOOL_UNAVAILABLE"})
# The codes a handler may raise; anything else becomes TOOL_FAILED.
HANDLER_CODES = frozenset({"TOOL_UNAVAILABLE", "MISSING_SCOPE", "TOOL_FAILED", "INVALID_ARGUMENTS"})
_IDENTITY_FIELDS = ("tenant_id", "school_id", "user_id", "student_id")


def message_for(code: str) -> str:
    return MESSAGES.get(code, MESSAGES["TOOL_FAILED"])


class ToolFailure(Exception):
    """Raised by a handler to fail with a code. Its text is never read."""

    def __init__(self, code: str = "TOOL_FAILED") -> None:
        super().__init__(code)
        self.code = code if code in HANDLER_CODES else "TOOL_FAILED"


class RunControl:
    """One deadline and one cancellation for a whole run, passed to every
    model call and tool call so they all stop together.

    The clock is injected, so tests and an on-device runtime control time."""

    def __init__(
        self,
        deadline: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._deadline = deadline
        self.clock = clock
        self._cancelled = asyncio.Event()
        self.reason: Literal["user", "parent", "shutdown", "deadline"] | None = None

    @classmethod
    def from_epoch(
        cls,
        deadline_epoch: float | None,
        *,
        wall: Callable[[], float] = time.time,
        clock: Callable[[], float] = time.monotonic,
    ) -> RunControl:
        """A control whose deadline is a wall-clock epoch, as a task carries."""
        if deadline_epoch is None:
            return cls(None, clock=clock)
        return cls(clock() + (deadline_epoch - wall()), clock=clock)

    def cancel(self, reason: Literal["user", "parent", "shutdown", "deadline"] = "user") -> None:
        if not self._cancelled.is_set():
            self.reason = reason
            self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    async def wait_cancelled(self) -> None:
        await self._cancelled.wait()

    def remaining(self) -> float | None:
        if self._deadline is None:
            return None
        return self._deadline - self.clock()

    def expired(self) -> bool:
        remaining = self.remaining()
        return remaining is not None and remaining <= 0

    def binds(self, seconds: float) -> bool:
        """Whether the run's deadline, not `seconds`, is what will stop a call."""
        remaining = self.remaining()
        return remaining is not None and remaining <= seconds

    def tighten(self, seconds: float) -> float:
        """The tighter of `seconds` and what remains of the deadline."""
        remaining = self.remaining()
        return seconds if remaining is None else max(0.0, min(seconds, remaining))


class ToolAuditRecord(BaseModel):
    """One record per call, however it ended. Ids, digests, codes and counts:
    never the arguments, the result or any passage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str
    execution_id: str
    tenant_id: str
    user_id: str
    tool: str
    tool_version: str | None
    side_effect: str | None
    args_digest: str
    idempotency_key: str | None
    status: Literal["ok", "replayed", "denied", "invalid", "failed", "timeout", "cancelled"]
    error_code: str | None
    retries: int
    replayed: bool
    input_bytes: int
    output_bytes: int
    sources_returned: int
    model_tokens: int
    latency_ms: float
    at: float
    boundaries: tuple[str, ...] = ()


class AuditSink(Protocol):
    def record(self, record: ToolAuditRecord) -> None: ...


class InMemoryAuditSink:
    def __init__(self) -> None:
        self.records: list[ToolAuditRecord] = []

    def record(self, record: ToolAuditRecord) -> None:
        self.records.append(record)


class ToolOutcome(BaseModel):
    """What a call returned to its caller."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str
    tool: str
    ok: bool
    data: dict[str, Any] | None = None
    error_code: str | None = None
    message: str | None = None
    retryable: bool = False
    replayed: bool = False
    retries: int = 0
    model_tokens: int = 0
    # False when the call stopped in flight: its effect, if any, is unknown,
    # and a repeat with the same key is what settles it.
    effect_known: bool = True
    audit: ToolAuditRecord | None = Field(default=None, exclude=True)

    def failure(self) -> dict[str, Any]:
        """The structured failure a model reads (the shape of #57)."""
        return {
            "ok": False,
            "error": {
                "code": self.error_code,
                "retryable": self.retryable,
                "message": self.message,
            },
        }


@dataclass(frozen=True)
class ToolCall:
    context: CallerContext
    args: dict[str, Any]
    services: ToolServices
    call_id: str
    execution_id: str
    idempotency_key: str | None = None
    # What the backend put in the signed request for this run.
    supplied: SuppliedData = field(default_factory=SuppliedData)


ToolHandler = Callable[[ToolCall], Awaitable[dict[str, Any]]]


@dataclass
class _Entry:
    args_digest: str
    outcome: ToolOutcome


class InMemoryIdempotencyStore:
    """First outcome per (scope, key). Scope names the caller and the run, so
    one student's key can never answer for another's."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def lock(self, scope: str) -> asyncio.Lock:
        return self._locks.setdefault(scope, asyncio.Lock())

    def get(self, scope: str) -> _Entry | None:
        return self._entries.get(scope)

    def put(self, scope: str, entry: _Entry) -> None:
        self._entries[scope] = entry


@dataclass(frozen=True)
class RegistryPolicy:
    # The ai-service default: nothing here changes state itself. A deployment
    # that adds backend write operations turns this on, deliberately.
    allow_write_through_backend: bool = False
    allow_proposals: bool = True


@dataclass
class _Registered:
    contract: ToolContract
    handler: ToolHandler


@dataclass
class _Meta:
    key: str | None
    input_bytes: int = 0


@dataclass
class _Attempt:
    outcome: ToolOutcome
    status: str = "ok"
    sources: int = 0
    output_bytes: int = 0


class ToolRegistry:
    def __init__(
        self,
        ports: Ports | None = None,
        *,
        policy: RegistryPolicy | None = None,
        audit: AuditSink | None = None,
        idempotency: InMemoryIdempotencyStore | None = None,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self._ports = ports or Ports()
        self._policy = policy or RegistryPolicy()
        self.audit: AuditSink = audit or InMemoryAuditSink()
        self._idempotency = idempotency or InMemoryIdempotencyStore()
        self._wall = wall
        self._tools: dict[str, _Registered] = {}

    # --- registration -------------------------------------------------------

    def register(self, contract: ToolContract, handler: ToolHandler) -> None:
        if contract.name in self._tools:
            raise ValueError("a tool with this name is already registered")
        if not asyncio.iscoroutinefunction(handler):
            raise TypeError("a tool handler is an async function")
        self._tools[contract.name] = _Registered(contract, handler)

    def contract(self, name: str) -> ToolContract | None:
        entry = self._tools.get(name)
        return entry.contract if entry else None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def available(
        self, context: CallerContext, allowed: tuple[str, ...] | frozenset[str] | None = None
    ) -> list[ToolContract]:
        """The contracts this caller may use: allowed by the task, permitted by
        the context, and of a side-effect class this registry runs."""
        out = []
        for name in self.names():
            contract = self._tools[name].contract
            if allowed is not None and name not in allowed:
                continue
            if self._refusal(contract, context) is None:
                out.append(contract)
        return out

    def _refusal(self, contract: ToolContract, context: CallerContext) -> str | None:
        if context.official_attempt_active:
            return "EXAM_MODE_ACTIVE"
        if not set(contract.required_permissions) <= set(context.permissions):
            return "PERMISSION_DENIED"
        if not set(contract.required_capabilities) <= set(context.capabilities):
            return "CAPABILITY_DENIED"
        if contract.side_effect is SideEffect.WRITE_THROUGH_BACKEND and (
            not self._policy.allow_write_through_backend
        ):
            return "SIDE_EFFECT_DISALLOWED"
        if contract.side_effect is SideEffect.PROPOSE and not self._policy.allow_proposals:
            return "SIDE_EFFECT_DISALLOWED"
        return None

    # --- execution ----------------------------------------------------------

    async def execute(
        self,
        name: str,
        args: dict[str, Any],
        context: CallerContext,
        *,
        call_id: str,
        execution_id: str,
        control: RunControl | None = None,
        allowed: tuple[str, ...] | frozenset[str] | None = None,
        idempotency_key: str | None = None,
        supplied: SuppliedData | None = None,
    ) -> ToolOutcome:
        started = time.monotonic()
        control = control or RunControl()
        entry = self._tools.get(name)
        contract = entry.contract if entry else None
        try:
            args_digest = digest(args)
        except (TypeError, ValueError):
            args_digest = "0" * 64
        meta = _Meta(key=idempotency_key)

        def done(attempt: _Attempt) -> ToolOutcome:
            record = ToolAuditRecord(
                call_id=call_id,
                execution_id=execution_id,
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                tool=name[:64],
                tool_version=contract.version if contract else None,
                side_effect=contract.side_effect.value if contract else None,
                args_digest=args_digest,
                idempotency_key=meta.key,
                status=attempt.status,  # type: ignore[arg-type]
                error_code=attempt.outcome.error_code,
                retries=attempt.outcome.retries,
                replayed=attempt.outcome.replayed,
                input_bytes=meta.input_bytes,
                output_bytes=attempt.output_bytes,
                sources_returned=attempt.sources,
                model_tokens=attempt.outcome.model_tokens,
                latency_ms=round((time.monotonic() - started) * 1000.0, 2),
                at=self._wall(),
                boundaries=tuple(b.value for b in contract.boundaries) if contract else (),
            )
            try:
                self.audit.record(record)
            except Exception as exc:  # the call's outcome stands; the gap is logged
                logger.error("Tool audit sink failed: %s", describe(exc))
            return attempt.outcome.model_copy(update={"audit": record})

        def refuse(code: str, status: str = "denied") -> ToolOutcome:
            return done(
                _Attempt(
                    ToolOutcome(
                        call_id=call_id,
                        tool=name[:64],
                        ok=False,
                        error_code=code,
                        message=message_for(code),
                        retryable=code in TRANSIENT_CODES,
                    ),
                    status=status,
                )
            )

        if entry is None or contract is None:
            return refuse("TOOL_NOT_FOUND")
        if allowed is not None and name not in allowed:
            return refuse("TOOL_NOT_ALLOWED")
        denied = self._refusal(contract, context)
        if denied:
            return refuse(denied)
        if control.cancelled:
            return refuse("EXECUTION_CANCELLED", "cancelled")
        if control.expired():
            return refuse("DEADLINE_EXPIRED", "timeout")

        # Arguments: size, shape, and no other tenant or user.
        try:
            encoded = canonical_json(args)
        except (TypeError, ValueError):
            return refuse("INVALID_ARGUMENTS", "invalid")
        meta.input_bytes = len(encoded.encode("utf-8"))
        if meta.input_bytes > contract.max_input_bytes:
            return refuse("INVALID_ARGUMENTS", "invalid")
        authority = {
            "tenant_id": context.tenant_id,
            "school_id": context.tenant_id,
            "user_id": context.user_id,
            "student_id": context.user_id,
        }
        for key in _IDENTITY_FIELDS:
            if key in args and args[key] is not None and str(args[key]) != authority[key]:
                return refuse("TENANT_MISMATCH", "denied")
        if validate(contract.input_schema, args):
            return refuse("INVALID_ARGUMENTS", "invalid")

        keyed = contract.idempotency is Idempotency.KEYED
        if keyed:
            key = idempotency_key or digest([execution_id, name, args_digest])[:32]
            meta.key = key
            scope = f"{context.tenant_id}|{context.user_id}|{execution_id}|{name}|{key}"
            async with self._idempotency.lock(scope):
                prior = self._idempotency.get(scope)
                if prior is not None:
                    if prior.args_digest != args_digest:
                        return refuse("IDEMPOTENCY_CONFLICT", "denied")
                    again = prior.outcome.model_copy(
                        update={"call_id": call_id, "replayed": True, "audit": None}
                    )
                    return done(_Attempt(again, status="replayed"))
                attempt = await self._run(
                    entry, args, context, call_id, execution_id, key, control, supplied
                )
                # An outcome is recorded unless it stopped in flight or may be
                # tried again: those a repeat with the same key settles.
                if attempt.outcome.effect_known and not attempt.outcome.retryable:
                    self._idempotency.put(scope, _Entry(args_digest, attempt.outcome))
                return done(attempt)

        return done(
            await self._run(entry, args, context, call_id, execution_id, None, control, supplied)
        )

    async def _run(
        self,
        entry: _Registered,
        args: dict[str, Any],
        context: CallerContext,
        call_id: str,
        execution_id: str,
        key: str | None,
        control: RunControl,
        supplied: SuppliedData | None = None,
    ) -> _Attempt:
        contract = entry.contract
        retries_allowed = (
            contract.retry.max_retries
            if contract.retry.classification is RetryClass.TRANSIENT
            else 0
        )
        retries = 0
        tokens = 0
        while True:
            metered = MeteredModel(self._ports.model) if self._ports.model else None
            services = ToolServices(
                contract.boundaries,
                retrieval=self._ports.retrieval,
                backend=self._ports.backend,
                model=metered,
            )
            call = ToolCall(
                context, args, services, call_id, execution_id, key, supplied or SuppliedData()
            )
            result = await self._once(entry, call, control)
            tokens += metered.tokens if metered else 0
            outcome = result.outcome
            if (
                outcome.ok
                or not outcome.retryable
                or retries >= retries_allowed
                or control.cancelled
                or control.expired()
            ):
                return _Attempt(
                    outcome.model_copy(update={"retries": retries, "model_tokens": tokens}),
                    status=result.status,
                    sources=result.sources,
                    output_bytes=result.output_bytes,
                )
            retries += 1

    async def _once(self, entry: _Registered, call: ToolCall, control: RunControl) -> _Attempt:
        contract = entry.contract

        def failed(code: str, status: str = "failed", effect_known: bool = True) -> _Attempt:
            return _Attempt(
                ToolOutcome(
                    call_id=call.call_id,
                    tool=contract.name,
                    ok=False,
                    error_code=code,
                    message=message_for(code),
                    retryable=code in TRANSIENT_CODES,
                    effect_known=effect_known,
                ),
                status=status,
            )

        timeout = control.tighten(contract.deadline_seconds)
        run_bound = control.binds(contract.deadline_seconds)
        if timeout <= 0:
            return failed("DEADLINE_EXPIRED", "timeout")
        task = asyncio.ensure_future(entry.handler(call))
        waiter = asyncio.ensure_future(control.wait_cancelled())
        try:
            finished, _ = await asyncio.wait(
                {task, waiter}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            task.cancel()
            waiter.cancel()
            raise
        waiter.cancel()
        if task not in finished:
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
            if control.cancelled:
                return failed("EXECUTION_CANCELLED", "cancelled", effect_known=False)
            if run_bound:  # the run's deadline stopped it, not the tool's own
                return failed("DEADLINE_EXPIRED", "timeout", effect_known=False)
            return failed("TOOL_TIMEOUT", "timeout", effect_known=False)
        try:
            raw = task.result()
        except BoundaryViolation:
            logger.error("Tool %s reached for a boundary it did not declare", contract.name)
            return failed("TOOL_FAILED")
        except ToolFailure as exc:
            return failed(exc.code)
        except asyncio.CancelledError:
            return failed("EXECUTION_CANCELLED", "cancelled", effect_known=False)
        except Exception as exc:
            logger.error("Tool %s failed: %s", contract.name, describe(exc))
            return failed("TOOL_FAILED")

        try:
            output_bytes = len(canonical_json(raw).encode("utf-8"))
        except (TypeError, ValueError):
            return failed("TOOL_OUTPUT_INVALID")
        if (
            not isinstance(raw, dict)
            or output_bytes > contract.max_output_bytes
            or validate(contract.output_schema, raw)
        ):
            return failed("TOOL_OUTPUT_INVALID")
        sources = 0
        if contract.produces_evidence and isinstance(raw.get("passages"), list):
            sources = len(raw["passages"])
        return _Attempt(
            ToolOutcome(call_id=call.call_id, tool=contract.name, ok=True, data=raw),
            status="ok",
            sources=sources,
            output_bytes=output_bytes,
        )
