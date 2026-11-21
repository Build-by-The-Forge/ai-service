"""The only doors a tool has to data and to a model.

A tool never holds a database session, a connection string or a URL. It is
handed a ToolServices, whose three doors open only if the tool's contract
declared the matching boundary:

- retrieval: passages from the caller's authorised course material, through
  the retrieval engine, which applies the knowledge scope and the backend's
  denied documents (the same path as chat);
- backend: authorised operations of the signed backend client;
- model: the model interface (the gateway online, a local model on a device).

These are Protocols, so a server adapter (``app/services/agentic``) and a
device adapter satisfy them without this module importing either.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.core.agentic.contracts import IDENT, Boundary


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceRef(_Frozen):
    """One exact document version, as retrieval records it."""

    asset_id: str
    run_number: int = Field(ge=0)
    attempt: int = Field(ge=0)


class EvidencePassage(_Frozen):
    number: int = Field(ge=1)
    chunk_id: str
    document_id: str
    page_start: int | None = None
    page_end: int | None = None
    text: str
    run_number: int = Field(default=1, ge=0)
    attempt: int = Field(default=1, ge=0)

    @property
    def source(self) -> SourceRef:
        return SourceRef(
            asset_id=self.document_id, run_number=self.run_number, attempt=self.attempt
        )


class EvidenceSet(_Frozen):
    query: str
    passages: tuple[EvidencePassage, ...]
    # Every source version retrieval returned, from the recorder alone.
    sources_used: tuple[SourceRef, ...]


class EvidencePort(Protocol):
    async def gather(
        self, query: str, *, tenant_id: str, denied_assets: frozenset[str]
    ) -> EvidenceSet: ...


class ProgressChange(_Frozen):
    """A change to planner state a student's evidence supports. Progress here is
    plan items done, topics reviewed and practice taken: never mastery and never
    a score of record, which the backend computes from verified attempts."""

    kind: Literal["plan_item_done", "topic_reviewed", "practice_taken"]
    subject_id: str = Field(pattern=IDENT.pattern)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=20)


class ProposalReceipt(_Frozen):
    proposal_id: str
    # The backend decides: accepted into its queue, or refused and dropped.
    status: Literal["queued", "refused"]


class BackendPort(Protocol):
    async def submit_progress_proposal(
        self,
        *,
        tenant_id: str,
        user_id: str,
        idempotency_key: str,
        change: ProgressChange,
    ) -> ProposalReceipt: ...


class ModelPort(Protocol):
    """The model door a tool uses. Deliberately not named like the gateway's
    interface (generate, generate_json): GatewayLLM is the only model interface
    in the app, and ModelCallerAdapter (app/services/agentic) adapts it to this."""

    async def complete(self, prompt: str) -> str: ...

    async def complete_json(self, prompt: str, *, kind: str, **hints: Any) -> dict[str, Any]: ...


def estimate_tokens(text: str) -> int:
    """The estimate used where the model interface reports no usage: about four
    characters a token, rounded up. The gateway still enforces the real budget."""
    return math.ceil(len(text) / 4)


class MeteredModel:
    """A ModelPort that counts the tokens it is asked for and returns, so a tool's
    nested model calls are charged to the run's budget."""

    def __init__(self, inner: ModelPort) -> None:
        self._inner = inner
        self.tokens = 0

    async def complete(self, prompt: str) -> str:
        answer = await self._inner.complete(prompt)
        self.tokens += estimate_tokens(prompt) + estimate_tokens(answer)
        return answer

    async def complete_json(self, prompt: str, *, kind: str, **hints: Any) -> dict[str, Any]:
        answer = await self._inner.complete_json(prompt, kind=kind, **hints)
        self.tokens += estimate_tokens(prompt) + estimate_tokens(repr(answer))
        return answer


class BoundaryViolation(Exception):
    """A tool reached for a door its contract did not declare."""


class ToolServices:
    """The doors, gated by the tool's declared boundaries."""

    def __init__(
        self,
        boundaries: tuple[Boundary, ...],
        *,
        retrieval: EvidencePort | None = None,
        backend: BackendPort | None = None,
        model: ModelPort | None = None,
    ) -> None:
        self._boundaries = frozenset(boundaries)
        self._retrieval = retrieval
        self._backend = backend
        self._model = model

    def _door(self, boundary: Boundary, door: Any) -> Any:
        if boundary not in self._boundaries or door is None:
            raise BoundaryViolation(boundary.value)
        return door

    @property
    def retrieval(self) -> EvidencePort:
        return self._door(Boundary.RETRIEVAL, self._retrieval)  # type: ignore[no-any-return]

    @property
    def backend(self) -> BackendPort:
        return self._door(Boundary.BACKEND, self._backend)  # type: ignore[no-any-return]

    @property
    def model(self) -> ModelPort:
        return self._door(Boundary.MODEL, self._model)  # type: ignore[no-any-return]


class Ports:
    """What a registry is built with: the real doors, before any tool's
    boundaries narrow them."""

    def __init__(
        self,
        *,
        retrieval: EvidencePort | None = None,
        backend: BackendPort | None = None,
        model: ModelPort | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.backend = backend
        self.model = model
