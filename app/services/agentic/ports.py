"""The retrieval door of the agentic tools, over the existing evidence boundary.

It adds nothing to retrieval: it calls ``app.services.rag.evidence.gather``, so
the tenant filter, the knowledge scope, the backend's denied documents and the
recorder (no passage without its recorded source version) apply exactly as they
do for chat.
"""

from typing import Any

from app.core.agentic.ports import EvidencePassage, EvidenceSet, SourceRef
from app.core.models.contracts import ModelInvocationContext
from app.services.ai.gateway_llm import ModelCaller
from app.services.rag.evidence import gather
from app.services.rag.retrieval import HybridRetrievalEngine


class RetrievalEvidencePort:
    def __init__(self, engine: HybridRetrievalEngine | None = None) -> None:
        self._engine = engine

    async def gather(
        self, query: str, *, tenant_id: str, denied_assets: frozenset[str]
    ) -> EvidenceSet:
        evidence = await gather(query, tenant_id, denied_assets, self._engine)
        return EvidenceSet(
            query=query,
            passages=tuple(
                EvidencePassage(
                    number=p.number,
                    chunk_id=p.chunk_id,
                    document_id=p.document_id,
                    page_start=p.page_start,
                    page_end=p.page_end,
                    text=p.text,
                    run_number=p.run_number,
                    attempt=p.attempt,
                )
                for p in evidence.passages
            ),
            sources_used=tuple(
                SourceRef(asset_id=s.asset_id, run_number=s.run_number, attempt=s.attempt)
                for s in evidence.sources_used
            ),
        )


class ModelCallerAdapter:
    """The model door over the gateway's one model interface.

    Each call binds the caller to the run's capability and invocation, so the
    gateway's approval check, deadline, budget and sanitised errors apply to a
    tool's nested model call as they do to chat. The capability is a decision
    for the reviewer: only the four chat-era capabilities have budgets today."""

    def __init__(
        self, caller: ModelCaller, capability: str, invocation: ModelInvocationContext | None
    ) -> None:
        self._caller = caller
        self._capability = capability
        self._invocation = invocation

    async def complete(self, prompt: str) -> str:
        with self._caller.for_capability(self._capability, self._invocation) as bound:
            return await bound.generate(prompt)

    async def complete_json(self, prompt: str, *, kind: str, **hints: Any) -> dict[str, Any]:
        with self._caller.for_capability(self._capability, self._invocation) as bound:
            return await bound.generate_json(prompt, kind=kind, **hints)
