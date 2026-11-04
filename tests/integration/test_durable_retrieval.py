"""RetrievalService over the durable index (P3-5): what ingestion indexed
is what the app retrieves, scoped to the verified tenant, with its page."""

import os
import uuid

import pytest

from app.core import tenant
from app.db.session import get_async_session_local, initialize_database, reset_engine
from app.services.ingestion.chunking import Chunk, chunk_id
from app.services.rag.embedder import MockEmbeddingAdapter
from app.services.rag.pg_index import PgChunkIndex
from app.services.rag.recorder import SourceUsed
from app.services.rag.retrieval import HybridRetrievalEngine, RetrievalService
from tests.rag_support import stored_scope

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"), reason="needs PostgreSQL"
)
TEXT = "Millikan's oil drop experiment determined the charge of the electron."


@pytest.fixture
async def durable():
    await initialize_database()
    index = PgChunkIndex(get_async_session_local())
    embedder = MockEmbeddingAdapter(dimension=384)
    yield index, embedder, HybridRetrievalEngine(embedder, index, index)
    await reset_engine()


async def retrieve_as(engine, tenant_id: str, query: str) -> list[dict]:
    tenant.set_request_scope(tenant_id)
    try:
        return (await RetrievalService(engine).retrieve(query, frozenset())).items
    finally:
        tenant.clear_request_scope()


async def test_indexed_chunks_are_retrieved_with_their_page_for_their_tenant_only(durable):
    index, embedder, engine = durable
    owner, other = str(uuid.uuid4()), str(uuid.uuid4())
    await index.raise_fence(tenant_id=owner, asset_id="asset-phy102", run_number=1, attempt=1)
    await index.replace_asset(
        tenant_id=owner,
        asset_id="asset-phy102",
        run_number=1,
        attempt=1,
        chunks=[Chunk(chunk_id(owner, "asset-phy102", 1, 0), 0, TEXT, 4, 4)],
        embeddings=[await embedder.embed_text(TEXT)],
        embedding_model="mock",
    )
    await stored_scope(owner, "asset-phy102")

    [hit] = await retrieve_as(engine, owner, "charge of the electron")
    assert hit["text"].startswith("Millikan's oil drop")
    assert hit["provenance"]["page_number"] == 4
    assert hit["provenance"]["source_uri"] == "asset:asset-phy102"

    assert await retrieve_as(engine, other, "charge of the electron") == []
    await index.delete_document(owner, "asset-phy102")


async def test_sources_used_names_the_run_and_attempt_that_were_retrieved(durable):
    index, embedder, engine = durable
    owner = str(uuid.uuid4())
    asset = f"asset-{uuid.uuid4().hex[:12]}"
    # Attempt 2 of run 3 wrote the index.
    await index.raise_fence(tenant_id=owner, asset_id=asset, run_number=3, attempt=2)
    await index.replace_asset(
        tenant_id=owner,
        asset_id=asset,
        run_number=3,
        attempt=2,
        chunks=[Chunk(chunk_id(owner, asset, 3, 0), 0, TEXT, 1, 1)],
        embeddings=[await embedder.embed_text(TEXT)],
        embedding_model="mock",
    )
    await stored_scope(owner, asset)
    tenant.set_request_scope(owner)
    try:
        found = await RetrievalService(engine).retrieve("charge of the electron", frozenset())
        evidence = await RetrievalService(engine).evidence("charge of the electron", frozenset())
    finally:
        tenant.clear_request_scope()
    assert found.sources_used == evidence.sources_used == (SourceUsed(asset, 3, 2),)


async def test_a_citation_names_the_run_and_attempt_whose_chunk_it_quotes(durable):
    from app.services.rag.evidence import check_grounded

    index, embedder, engine = durable
    owner = str(uuid.uuid4())
    asset = f"asset-{uuid.uuid4().hex[:12]}"
    await index.raise_fence(tenant_id=owner, asset_id=asset, run_number=4, attempt=2)
    await index.replace_asset(
        tenant_id=owner,
        asset_id=asset,
        run_number=4,
        attempt=2,
        chunks=[Chunk(chunk_id(owner, asset, 4, 0), 0, TEXT, 6, 6)],
        embeddings=[await embedder.embed_text(TEXT)],
        embedding_model="mock",
    )
    await stored_scope(owner, asset)
    tenant.set_request_scope(owner)
    try:
        evidence = await RetrievalService(engine).evidence("charge of the electron", frozenset())
    finally:
        tenant.clear_request_scope()
    [citation] = check_grounded("Millikan measured it [1].", evidence).citations
    assert (citation["document_id"], citation["run_number"], citation["attempt"]) == (asset, 4, 2)
    assert citation["excerpt"] == TEXT and citation["page_start"] == 6
