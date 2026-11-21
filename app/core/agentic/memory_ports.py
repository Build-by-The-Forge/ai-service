"""In-memory reference implementations of the ports.

Deterministic and dependency-free: for tests, the evaluation harness and as the
shape a device implementation follows (its corpus is the active content pack, its
backend a queue synced later). They apply the same rules the server does: a
tenant sees only its own passages, a denied document is never returned, and a
proposal key is honoured once.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.core.agentic.ports import (
    EvidencePassage,
    EvidenceSet,
    ProgressChange,
    ProposalReceipt,
    SourceRef,
)

_WORD = re.compile(r"[a-z0-9]{3,}")
TOP_K = 5


@dataclass(frozen=True)
class CorpusPassage:
    tenant_id: str
    chunk_id: str
    document_id: str
    text: str
    page: int | None = None
    run_number: int = 1
    attempt: int = 1


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


class InMemoryEvidencePort:
    """Word-overlap retrieval over a fixed corpus, with the scope rules of the
    real engine: tenant first, then denied documents."""

    def __init__(self, corpus: list[CorpusPassage]) -> None:
        self._corpus = list(corpus)
        self.calls: list[tuple[str, str, frozenset[str]]] = []

    async def gather(
        self, query: str, *, tenant_id: str, denied_assets: frozenset[str]
    ) -> EvidenceSet:
        self.calls.append((query, tenant_id, denied_assets))
        wanted = _words(query)
        scored = []
        for item in self._corpus:
            if item.tenant_id != tenant_id or item.document_id in denied_assets:
                continue
            overlap = len(wanted & _words(item.text))
            if overlap:
                scored.append((-overlap, item.chunk_id, item))
        scored.sort(key=lambda row: (row[0], row[1]))
        passages = tuple(
            EvidencePassage(
                number=n,
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                page_start=item.page,
                page_end=item.page,
                text=item.text,
                run_number=item.run_number,
                attempt=item.attempt,
            )
            for n, (_, _, item) in enumerate(scored[:TOP_K], start=1)
        )
        sources = tuple(sorted({p.source for p in passages}, key=lambda s: s.model_dump_json()))
        return EvidenceSet(query=query, passages=passages, sources_used=sources)


class InMemoryBackendPort:
    """Records proposals. The same idempotency key returns the first receipt, as
    the backend's operation would, so a retried call has no second effect."""

    def __init__(self, *, refuse: bool = False) -> None:
        self.proposals: list[tuple[str, str, str, ProgressChange]] = []
        self._by_key: dict[str, ProposalReceipt] = {}
        self._refuse = refuse
        self.requests = 0

    async def submit_progress_proposal(
        self,
        *,
        tenant_id: str,
        user_id: str,
        idempotency_key: str,
        change: ProgressChange,
    ) -> ProposalReceipt:
        self.requests += 1
        scoped = f"{tenant_id}|{user_id}|{idempotency_key}"
        if scoped in self._by_key:
            return self._by_key[scoped]
        proposal_id = "prop_" + hashlib.sha256(scoped.encode()).hexdigest()[:12]
        receipt = ProposalReceipt(
            proposal_id=proposal_id, status="refused" if self._refuse else "queued"
        )
        self._by_key[scoped] = receipt
        self.proposals.append((tenant_id, user_id, idempotency_key, change))
        return receipt


def sources_of(passages: tuple[EvidencePassage, ...]) -> tuple[SourceRef, ...]:
    return tuple(sorted({p.source for p in passages}, key=lambda s: s.model_dump_json()))
