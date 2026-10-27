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
from app.services.rag.retrieval import HybridRetrievalEngine, RetrievalService

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
        return await RetrievalService(engine).retrieve(query)
    finally:
        tenant.clear_request_scope()


async def test_indexed_chunks_are_retrieved_with_their_page_for_their_tenant_only(durable):
    index, embedder, engine = durable
    owner, other = str(uuid.uuid4()), str(uuid.uuid4())
    await index.replace_asset(
        tenant_id=owner,
        asset_id="asset-phy102",
        run_number=1,
        chunks=[Chunk(chunk_id(owner, "asset-phy102", 1, 0), 0, TEXT, 4, 4)],
        embeddings=[await embedder.embed_text(TEXT)],
        embedding_model="mock",
    )

    [hit] = await retrieve_as(engine, owner, "charge of the electron")
    assert hit["text"].startswith("Millikan's oil drop")
    assert hit["provenance"]["page_number"] == 4
    assert hit["provenance"]["source_uri"] == "asset:asset-phy102"

    assert await retrieve_as(engine, other, "charge of the electron") == []
    await index.delete_document(owner, "asset-phy102")
