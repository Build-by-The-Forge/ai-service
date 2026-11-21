"""The contracts bounded agentic study workflows share, online and on a device.

Three models, and what travels with them:

- ToolContract: what a tool is, and what it may do. Names, schemas, the
  permission it needs, whether repeating a call is safe, how long it may take,
  what may be retried and what it can change.
- AgentTask: what a run is asked to do, for whom, with which tools, evidence,
  budget and deadline, and when it counts as complete.
- ExecutionCheckpoint: where a run is, in enough detail to resume it, audit
  it or stop it: step, calls, budget used, retries, state, cancellation.

Rules these models keep:

- Pydantic, strict about extra fields, and versioned: every top-level model
  has ``schema_version``, and a version this code does not know is refused.
- JSON in, JSON out. ``model_dump(mode="json")`` and ``model_validate_json``
  round-trip, so the same record can be stored on a server, in a device's
  database or in a test file. No sets, no datetimes, no callables.
- No provider types, no server types. Nothing here names a model provider or
  imports the web framework, the database, the gateway or any other module of
  the app. A device implementation needs only this file and ``schema.py``.
- Caller authority (tenant, user, permissions, denied documents, whether an
  official attempt is under way) arrives in ``CallerContext``, built by the
  trusted caller from the verified request. Nothing a model writes can set it.

Where this is server-shaped, and the device must do its own thing: the deadline
is a wall-clock epoch (``deadline_epoch``) because it must cross processes and
survive a restart; a device with a lease clock converts it as it needs.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.agentic.schema import SchemaError, check_schema

SCHEMA_VERSION: Literal[1] = 1

TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
PERMISSION = re.compile(r"^[a-z][a-z_]*:[a-z][a-z_]*$")
CAPABILITY = re.compile(r"^[a-z][a-z0-9_:.]{1,63}$")
SEMVER = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}$")
IDENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class _Model(BaseModel):
    """Common rules: no unknown fields, immutable once built."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=False)


def canonical_json(value: Any) -> str:
    """One JSON text per value: sorted keys, no spaces, NaN and Infinity
    refused. Digests, idempotency keys and replay all hash this."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


# --- Tool contract -----------------------------------------------------------


class SideEffect(StrEnum):
    """What a tool can change. The registry refuses a tool of a class it was not
    configured to run."""

    READ = "read"  # changes nothing
    PROPOSE = "propose"  # returns something for BUILD or the student to approve; changes nothing
    WRITE_THROUGH_BACKEND = (
        "write_through_backend"  # asks an authorised backend operation to change state
    )


class Idempotency(StrEnum):
    NATURAL = "natural"  # repeating the call is harmless
    KEYED = "keyed"  # needs a key; the same key returns the first outcome, never a second effect
    NONE = "none"  # repeating is unsafe: never retried or replayed by the runtime


class RetryClass(StrEnum):
    NEVER = "never"
    TRANSIENT = (
        "transient"  # only a failure the tool marks transient (a timeout, an unavailable port)
    )


class Boundary(StrEnum):
    """The only ways a tool reaches data or a model. There is no database
    boundary: a tool holds no connection and no session."""

    RETRIEVAL = "retrieval"  # the retrieval engine, with the caller's scope and denied documents
    BACKEND = "backend"  # the signed backend client's authorised operations
    MODEL = "model"  # the model interface (the gateway, or a local model)


class RetryPolicy(_Model):
    classification: RetryClass = RetryClass.NEVER
    max_retries: int = Field(default=0, ge=0, le=3)

    @model_validator(mode="after")
    def _consistent(self) -> RetryPolicy:
        if self.classification is RetryClass.NEVER and self.max_retries:
            raise ValueError("a tool that is never retried has no retries")
        if self.classification is RetryClass.TRANSIENT and not self.max_retries:
            raise ValueError("a transient retry class needs max_retries")
        return self


class ToolContract(_Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    name: str
    version: str
    description: str = Field(min_length=10, max_length=600)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    required_permissions: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    side_effect: SideEffect = SideEffect.READ
    idempotency: Idempotency = Idempotency.NATURAL
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    deadline_seconds: float = Field(default=15.0, gt=0, le=120)
    boundaries: tuple[Boundary, ...] = ()
    # The tool's passages are evidence: the runtime records them, and an
    # answer may cite only what such a tool returned.
    produces_evidence: bool = False
    max_input_bytes: int = Field(default=16_384, ge=64, le=262_144)
    max_output_bytes: int = Field(default=65_536, ge=64, le=1_048_576)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not TOOL_NAME.match(value):
            raise ValueError("tool names are lower_snake_case, 3 to 64 characters")
        return value

    @field_validator("version")
    @classmethod
    def _version(cls, value: str) -> str:
        if not SEMVER.match(value):
            raise ValueError("version is MAJOR.MINOR.PATCH")
        return value

    @field_validator("required_permissions")
    @classmethod
    def _permissions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not PERMISSION.match(p) for p in value) or len(set(value)) != len(value):
            raise ValueError("permissions are unique 'area:action' names")
        return value

    @field_validator("required_capabilities")
    @classmethod
    def _capabilities(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not CAPABILITY.match(c) for c in value) or len(set(value)) != len(value):
            raise ValueError("capabilities are unique names")
        return value

    @field_validator("boundaries")
    @classmethod
    def _boundaries(cls, value: tuple[Boundary, ...]) -> tuple[Boundary, ...]:
        if len(set(value)) != len(value):
            raise ValueError("boundaries are unique")
        return value

    @field_validator("input_schema", "output_schema")
    @classmethod
    def _schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            check_schema(value)
        except SchemaError as exc:
            raise ValueError(str(exc)) from None
        if value.get("type") != "object":
            raise ValueError("a tool's input and output are objects")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> ToolContract:
        if self.input_schema.get("additionalProperties") is not False:
            # A model must not smuggle fields (a tenant, a score) past the schema.
            raise ValueError("input_schema must set additionalProperties to false")
        if self.side_effect is SideEffect.READ and self.idempotency is not Idempotency.NATURAL:
            raise ValueError("a read is naturally idempotent")
        if self.side_effect is not SideEffect.READ and self.idempotency is not Idempotency.KEYED:
            raise ValueError(
                "a tool that proposes or writes is keyed, so a repeat has no second effect"
            )
        if (
            self.idempotency is Idempotency.NONE
            and self.retry.classification is not RetryClass.NEVER
        ):
            raise ValueError("a tool whose repeat is unsafe is never retried")
        if (
            self.side_effect is SideEffect.WRITE_THROUGH_BACKEND
            and Boundary.BACKEND not in self.boundaries
        ):
            raise ValueError("a write goes through the backend boundary")
        if self.produces_evidence and Boundary.RETRIEVAL not in self.boundaries:
            raise ValueError("evidence comes from the retrieval boundary")
        return self

    def describe_for_model(self) -> dict[str, Any]:
        """What the model sees of a tool: provider-neutral, no permissions."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# --- Caller authority --------------------------------------------------------


class CallerContext(_Model):
    """Who a run acts for, from the verified request. The model never sets it."""

    tenant_id: str = Field(pattern=IDENT.pattern)
    user_id: str = Field(pattern=IDENT.pattern)
    permissions: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    # The backend's denied documents. None means the run did not come from a
    # signed backend call, and retrieval refuses (as content.search does).
    denied_assets: tuple[str, ...] | None = None
    # Supplied by the backend. While an official attempt is active, no agent
    # runs for this user.
    official_attempt_active: bool = False

    @field_validator("permissions", "capabilities", "denied_assets")
    @classmethod
    def _unique(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("entries are unique")
        return value


# --- Agent task --------------------------------------------------------------


class ExecutionMode(StrEnum):
    ONLINE = "online"
    ON_DEVICE = "on_device"


class AgentBudget(_Model):
    """Hard ceilings. A step is one model decision; an iteration is one tool
    round (a tool call the model asked for). Tokens are estimated when the
    model interface reports none (see AgentModel)."""

    max_steps: int = Field(default=8, ge=1, le=50)
    max_iterations: int = Field(default=6, ge=0, le=50)
    max_tokens: int = Field(default=8_000, ge=1, le=200_000)
    max_cost_usd: float | None = Field(default=None, ge=0)
    max_retries: int = Field(default=3, ge=0, le=20)


class ExecutionPolicy(_Model):
    mode: ExecutionMode = ExecutionMode.ONLINE
    # The same call (tool and arguments) this many times is a loop.
    doom_loop_repeat_limit: int = Field(default=3, ge=2, le=10)
    # This many tool rounds in a row that add nothing new is a loop.
    no_progress_limit: int = Field(default=3, ge=2, le=10)
    # Model failures or malformed decisions in a row before the run fails.
    max_consecutive_failures: int = Field(default=2, ge=1, le=5)
    step_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    # On budget exhaustion, return the tool results gathered (marked partial).
    allow_partial_result: bool = True


class EvidenceRequirements(_Model):
    """What an answer must rest on."""

    require_sources: bool = True
    min_sources: int = Field(default=1, ge=0, le=20)
    # When the evidence is not enough, abstain rather than answer.
    abstain_if_insufficient: bool = True

    @model_validator(mode="after")
    def _consistent(self) -> EvidenceRequirements:
        if self.require_sources and self.min_sources < 1:
            raise ValueError("requiring sources needs min_sources of at least 1")
        return self


class CompletionCriteria(_Model):
    """When a run counts as done: a final answer, and these tools having run."""

    require_final_answer: bool = True
    required_tools: tuple[str, ...] = ()

    @field_validator("required_tools")
    @classmethod
    def _tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not TOOL_NAME.match(t) for t in value) or len(set(value)) != len(value):
            raise ValueError("required tools are unique tool names")
        return value


class PracticeEvidence(_Model):
    """One practice result the backend supplied: a fact about what the student
    did, never a score of record."""

    topic_id: str = Field(pattern=IDENT.pattern)
    correct: bool
    # Epoch seconds, so recency can be weighed deterministically.
    answered_at: float = Field(ge=0)
    attempt_ref: str = Field(pattern=IDENT.pattern)


class TopicRef(_Model):
    topic_id: str = Field(pattern=IDENT.pattern)
    title: str = Field(min_length=1, max_length=200)


class PlanItem(_Model):
    item_id: str = Field(pattern=IDENT.pattern)
    topic_id: str = Field(pattern=IDENT.pattern)
    due_epoch_day: int = Field(ge=0)
    activity: Literal["review", "practice", "read"]
    minutes: int = Field(ge=5, le=240)
    done: bool = False


class SuppliedData(_Model):
    """What the backend put in the signed request for this run. Tools read
    these and nothing else about the student's history."""

    topics: tuple[TopicRef, ...] = Field(default=(), max_length=200)
    practice_evidence: tuple[PracticeEvidence, ...] = Field(default=(), max_length=2_000)
    plan: tuple[PlanItem, ...] = Field(default=(), max_length=500)
    # The student's exam date, as an epoch day, if they gave one.
    exam_epoch_day: int | None = Field(default=None, ge=0)
    today_epoch_day: int | None = Field(default=None, ge=0)


class AgentTask(_Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    task_id: str = Field(pattern=IDENT.pattern)
    execution_id: str = Field(pattern=IDENT.pattern)
    task_type: str = Field(pattern=r"^[a-z][a-z0-9_]{2,47}$")
    goal: str = Field(min_length=1, max_length=2_000)
    context: CallerContext
    required_capabilities: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    evidence: EvidenceRequirements = Field(default_factory=EvidenceRequirements)
    budget: AgentBudget = Field(default_factory=AgentBudget)
    # Wall-clock epoch seconds. One deadline for the whole run: every model
    # call and tool call inherits what remains of it.
    deadline_epoch: float | None = Field(default=None, gt=0)
    execution_policy: ExecutionPolicy = Field(default_factory=ExecutionPolicy)
    completion: CompletionCriteria = Field(default_factory=CompletionCriteria)
    supplied: SuppliedData = Field(default_factory=SuppliedData)

    @field_validator("allowed_tools")
    @classmethod
    def _tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not TOOL_NAME.match(t) for t in value) or len(set(value)) != len(value):
            raise ValueError("allowed tools are unique tool names")
        return value

    @field_validator("required_capabilities")
    @classmethod
    def _capabilities(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not CAPABILITY.match(c) for c in value) or len(set(value)) != len(value):
            raise ValueError("capabilities are unique names")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> AgentTask:
        missing = set(self.completion.required_tools) - set(self.allowed_tools)
        if missing:
            raise ValueError("a required tool must be an allowed tool")
        return self

    def digest(self) -> str:
        return digest(self.model_dump(mode="json"))


# --- Execution checkpoint ----------------------------------------------------


class ExecutionState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"  # a grounded, checked answer
    ABSTAINED = "abstained"  # the material did not support an answer
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    BUDGET_EXHAUSTED = "budget_exhausted"
    LOOP_DETECTED = "loop_detected"
    REFUSED = "refused"  # never started: exam mode, permission, capability


TERMINAL_STATES = frozenset(
    {
        ExecutionState.COMPLETED,
        ExecutionState.ABSTAINED,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
        ExecutionState.TIMED_OUT,
        ExecutionState.BUDGET_EXHAUSTED,
        ExecutionState.LOOP_DETECTED,
        ExecutionState.REFUSED,
    }
)
# Only a run that was interrupted (not one that ended) can resume.
RESUMABLE_STATES = frozenset({ExecutionState.RUNNING})

ALLOWED_TRANSITIONS: dict[ExecutionState, frozenset[ExecutionState]] = {
    ExecutionState.PENDING: frozenset({ExecutionState.RUNNING, ExecutionState.REFUSED}),
    ExecutionState.RUNNING: TERMINAL_STATES - {ExecutionState.REFUSED},
}


def can_transition(current: ExecutionState, new: ExecutionState) -> bool:
    return new in ALLOWED_TRANSITIONS.get(current, frozenset())


class CallKind(StrEnum):
    MODEL = "model"
    TOOL = "tool"


class CallStatus(StrEnum):
    STARTED = "started"  # written before the call, so a crash leaves a trace
    OK = "ok"
    FAILED = "failed"
    REPLAYED = "replayed"  # the first outcome returned again, no second effect
    DENIED = "denied"
    CANCELLED = "cancelled"  # stopped in flight: the effect, if any, is unknown


class CallRecord(_Model):
    call_id: str = Field(pattern=IDENT.pattern)
    step: int = Field(ge=0)
    kind: CallKind
    # The tool's name, or "decision" for a model call.
    name: str = Field(min_length=1, max_length=64)
    # A digest of the arguments, never the arguments.
    args_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    idempotency_key: str | None = Field(default=None, max_length=128)
    status: CallStatus = CallStatus.STARTED
    error_code: str | None = Field(default=None, max_length=64)
    retries: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)
    result_ref: str | None = Field(default=None, pattern=IDENT.pattern)


class ResultRef(_Model):
    """A pointer to a stored result, with its hash: the store can be
    anywhere (memory, a file, a device database) and a changed result shows."""

    ref_id: str = Field(pattern=IDENT.pattern)
    call_id: str = Field(pattern=IDENT.pattern)
    step: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BudgetConsumed(_Model):
    steps: int = Field(default=0, ge=0)
    iterations: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    retries: int = Field(default=0, ge=0)


class CancellationState(_Model):
    requested: bool = False
    reason: Literal["user", "parent", "shutdown", "deadline"] | None = None
    at_step: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> CancellationState:
        if self.requested != (self.reason is not None):
            raise ValueError("a cancellation has a reason, and only a cancellation does")
        return self


class ExecutionCheckpoint(_Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    execution_id: str = Field(pattern=IDENT.pattern)
    task_id: str = Field(pattern=IDENT.pattern)
    # The task this checkpoint belongs to: a resume with another task is refused.
    task_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    tenant_id: str = Field(pattern=IDENT.pattern)
    user_id: str = Field(pattern=IDENT.pattern)
    step: int = Field(default=0, ge=0)
    state: ExecutionState = ExecutionState.PENDING
    calls: tuple[CallRecord, ...] = ()
    results: tuple[ResultRef, ...] = ()
    budget_consumed: BudgetConsumed = Field(default_factory=BudgetConsumed)
    retry_count: int = Field(default=0, ge=0)
    cancellation: CancellationState = Field(default_factory=CancellationState)
    # Why the run ended, as a fixed code (never an exception's text).
    outcome_code: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _consistent(self) -> ExecutionCheckpoint:
        terminal = self.state in TERMINAL_STATES
        if terminal and self.state is not ExecutionState.COMPLETED and not self.outcome_code:
            raise ValueError("a run that did not complete says why")
        if not terminal and self.outcome_code:
            raise ValueError("a run still going has no outcome")
        if any(call.step > self.step for call in self.calls):
            raise ValueError("a call is from a step that has not happened")
        ids = [call.call_id for call in self.calls]
        if len(set(ids)) != len(ids):
            raise ValueError("call ids are unique")
        refs = {r.ref_id for r in self.results}
        if any(call.result_ref and call.result_ref not in refs for call in self.calls):
            raise ValueError("a call points at a result the checkpoint does not list")
        if self.cancellation.requested and self.state is ExecutionState.COMPLETED:
            raise ValueError("a cancelled run did not complete")
        return self
