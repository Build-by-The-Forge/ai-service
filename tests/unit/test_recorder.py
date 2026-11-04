"""The retrieval recorder (M3-D provenance): sources_used is what the store
returned, cut for length or not, and a prompt takes nothing it didn't record."""

import pytest

from app.core.rag.contracts import (
    DocumentChunk,
    RetrievalFilter,
    RetrievalPolicy,
    RetrievedChunk,
    RetrievedContext,
)
from app.services.rag.evidence import Evidence, Passage, gather
from app.services.rag.recorder import RetrievalRecorder, SourceUsed, UnrecordedPassageError
from tests.rag_support import in_memory_engine, institutional

TENANT = "tenant-a"


def chunk(
    asset: str, run: int = 1, attempt: int = 1, text: str = "Osmosis moves water."
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{asset}-{run}-{attempt}",
        tenant_id=TENANT,
        document_id=asset,
        document_title=asset,
        content=text,
        metadata={"run_number": run, "attempt": attempt, "page_start": 1, "page_end": 1},
    )


def test_sources_are_each_document_version_once_in_order():
    recorder = RetrievalRecorder()
    recorder.record([chunk("b", 2, 1), chunk("a"), chunk("b", 2, 1)])
    recorder.record([chunk("a", 1, 2)])
    assert recorder.sources_used() == (
        SourceUsed("a", 1, 1),
        SourceUsed("a", 1, 2),
        SourceUsed("b", 2, 1),
    )


def test_a_passage_the_recorder_did_not_record_is_refused():
    recorder = RetrievalRecorder()
    recorder.record([chunk("a")])
    assert recorder.passage("a-1-1").document_id == "a"
    with pytest.raises(UnrecordedPassageError):
        recorder.passage("b-1-1")


def test_evidence_refuses_a_passage_whose_document_is_not_a_source():
    passage = Passage(1, "text", "c1", "asset-x", 1, 1)
    with pytest.raises(UnrecordedPassageError):
        Evidence("q", (passage,), (SourceUsed("asset-y", 1, 1),))
    with pytest.raises(UnrecordedPassageError):
        Evidence("q", (passage,), ())


async def test_a_passage_cut_for_length_is_still_a_source(in_memory_scopes):
    engine = in_memory_engine(in_memory_scopes)
    for asset in ("a1", "a2", "a3"):
        await engine.index_document(
            tenant_id=TENANT,
            document_id=asset,
            document_title=asset,
            text=f"Osmosis moves water across a membrane, says {asset}.",
        )
    await institutional(in_memory_scopes, TENANT, "a1", "a2", "a3")
    recorder = RetrievalRecorder()
    context = await engine.retrieve(
        "osmosis water membrane",
        filter=RetrievalFilter(tenant_id=TENANT, denied_assets=frozenset()),
        # The store returns all three; the size limit keeps one for the prompt.
        policy=RetrievalPolicy(top_k=3, max_context_bytes=10),
        recorder=recorder,
    )
    assert len(context.chunks) == 1  # only one passage reaches the prompt
    assert {s.asset_id for s in recorder.sources_used()} == {"a1", "a2", "a3"}


async def test_gather_takes_passages_only_from_the_recorder():
    class Unrecorded:
        """An engine that hands back a chunk without recording it."""

        async def retrieve(self, query, filter, policy=None, recorder=None):
            return RetrievedContext(
                query=query, tenant_id=filter.tenant_id, chunks=[RetrievedChunk(chunk=chunk("a"))]
            )

    with pytest.raises(UnrecordedPassageError):
        await gather("osmosis", TENANT, frozenset(), engine=Unrecorded())


async def test_a_denied_document_is_not_a_source(in_memory_scopes):
    engine = in_memory_engine(in_memory_scopes)
    for asset in ("kept", "denied"):
        await engine.index_document(
            tenant_id=TENANT, document_id=asset, document_title=asset, text="Osmosis moves water."
        )
    await institutional(in_memory_scopes, TENANT, "kept", "denied")
    evidence = await gather("osmosis", TENANT, frozenset({"denied"}), engine=engine)
    assert [s.asset_id for s in evidence.sources_used] == ["kept"]
