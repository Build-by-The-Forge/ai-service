"""Ingestion end to end on the AI side (P3-4): a real PDF is fetched,
extracted, chunked, embedded by the real model and committed to the real
pgvector index; the job completes, and a question retrieves the chunk for
its tenant only.

Needs PostgreSQL with pgvector (DATABASE_URL) and the verified model files
(EMBEDDING_MODEL_DIR; the image sets it).
"""

import os
import uuid
from pathlib import Path

import pytest

from app.core.callbacks.dispatcher import backend_callback_body
from app.core.callbacks.events import CallbackEventType
from app.core.execution_context import TaskStatus
from app.core.rag.contracts import RetrievalFilter
from app.db.session import get_async_session_local, initialize_database, reset_engine
from app.services.rag.onnx_embedder import OnnxMiniLMEmbedder
from app.services.rag.pg_index import PgChunkIndex
from tests.unit.test_ingestion_fetch_lifecycle import (
    ASSET,
    KEY,
    PDF,
    TENANT,
    FakeS3,
    handler,
    payload,
)
from tests.unit.test_worker_callback_integration import kinds, run, task, worker_with

MODEL_DIR = os.getenv("EMBEDDING_MODEL_DIR")
pytestmark = [
    pytest.mark.skipif(
        not os.getenv("DATABASE_URL", "").startswith("postgresql"), reason="needs PostgreSQL"
    ),
    pytest.mark.skipif(
        not (MODEL_DIR and (Path(MODEL_DIR) / "model.onnx").is_file()),
        reason="needs the verified model files (EMBEDDING_MODEL_DIR)",
    ),
]


async def test_an_ingested_document_is_retrievable_by_its_tenant_only(tmp_path):
    await initialize_database()
    try:
        index = PgChunkIndex(get_async_session_local())
        embedder = OnnxMiniLMEmbedder(MODEL_DIR, threads=1)
        worker, transport, dispatcher = worker_with(
            handler(FakeS3({KEY: PDF}), tmp_path, embedder=embedder, index=index)
        )

        _, status = await run(worker, transport, task(payload=payload()))

        assert status == TaskStatus.SUCCEEDED
        assert kinds(dispatcher)[-2:] == [
            (CallbackEventType.STAGE_COMPLETED, "indexed"),
            (CallbackEventType.COMPLETED, None),
        ]
        indexed = backend_callback_body(dispatcher.delivered_events[-2])["metrics"]
        assert indexed["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"

        question = await embedder.embed_text("what are these lecture notes about? limits")
        mine = RetrievalFilter(tenant_id=TENANT)
        (dense, score), *_ = await index.query_dense(question, mine, top_k=3)
        (sparse, _), *_ = await index.query_sparse("notes on limits", mine, top_k=3)
        assert dense.document_id == sparse.document_id == ASSET and score > 0.6
        assert "limits" in dense.content and dense.provenance.page_number == 1

        other = RetrievalFilter(tenant_id=str(uuid.uuid4()))
        assert await index.query_dense(question, other, top_k=3) == []
        assert await index.query_sparse("notes on limits", other, top_k=3) == []
        await index.delete_document(TENANT, ASSET)
    finally:
        await reset_engine()
