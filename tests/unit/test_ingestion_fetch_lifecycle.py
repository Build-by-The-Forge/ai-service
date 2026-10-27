"""The ingestion handler in the worker lifecycle: the asset is fetched and
verified (P3-1), extracted (P3-2), chunked (P3-3), embedded and durably
indexed (P3-4), each stage reported with its evidence, and only then is the
job completed. A failure reports its own typed code with the evidence of
the stages before it, retried only if transient."""

import dataclasses
import hashlib

import fitz
import pytest
from botocore.exceptions import EndpointConnectionError

from app.core.callbacks.dispatcher import backend_callback_body
from app.core.callbacks.events import CallbackEventType
from app.core.execution_context import TaskStatus
from app.core.rag.errors import EmbeddingGenerationError, VectorStoreError
from app.services.ingestion.asset_fetcher import AssetFetcher
from app.services.ingestion.extraction import DocumentExtractor
from app.services.tasks.handlers import ContentIngestionTaskHandler
from app.services.tasks.ingestion_outcomes import PIPELINE_UNAVAILABLE
from tests.unit.test_asset_fetcher import FakeS3
from tests.unit.test_worker_callback_integration import kinds, run, task, worker_with

BUCKET = "build-documents"
TENANT = "6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c"  # the task's verified tenant
ASSET = "0b9a8c7d-6e5f-5a4b-8c3d-2e1f0a9b8c7d"


def _native_pdf() -> bytes:
    doc = fitz.open()
    doc.new_page().insert_textbox(fitz.Rect(50, 50, 550, 750), "Lecture notes on limits. " * 20)
    data = doc.tobytes()
    doc.close()
    return data


PDF = _native_pdf()
SHA = hashlib.sha256(PDF).hexdigest()
KEY = f"tenants/{TENANT}/assets/{ASSET}/{SHA}"


def payload(**overrides) -> dict:
    return {
        "asset_id": ASSET,
        "storage_uri": f"s3://{BUCKET}/{KEY}",
        "content_sha256": SHA,
        "mime_type": "application/pdf",
        "file_size_bytes": len(PDF),
    } | overrides


def _no_ocr():
    raise AssertionError("a native PDF needs no OCR engine")


class FakeEmbedder:
    dimension = 384
    model_id = "fake-minilm"

    def __init__(self, error: Exception | None = None):
        self.error = error

    async def embed_batch(self, texts):
        if self.error:
            raise self.error
        return [[1.0] + [0.0] * 383 for _ in texts]


class FakeIndex:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[dict] = []

    async def replace_asset(self, **kwargs):
        if self.error:
            raise self.error
        self.calls.append(kwargs)
        return len(kwargs["chunks"])


def handler(client, tmp_path, embedder=None, index=None) -> ContentIngestionTaskHandler:
    return ContentIngestionTaskHandler(
        AssetFetcher(
            bucket=BUCKET,
            client_factory=lambda: client,
            max_bytes=10 * 1024 * 1024,
            timeout_seconds=5,
            temp_dir=str(tmp_path),
        ),
        DocumentExtractor(
            ocr_engine_factory=_no_ocr, max_pages=10, timeout_seconds=30, min_confidence=0.5
        ),
        embedder or FakeEmbedder(),
        index if index is not None else FakeIndex(),
    )


async def test_every_stage_is_reported_and_only_then_the_job_completes(tmp_path):
    index = FakeIndex()
    worker, transport, dispatcher = worker_with(handler(FakeS3({KEY: PDF}), tmp_path, index=index))

    msg_id, status = await run(worker, transport, task(payload=payload(run_number=2)))

    assert status == TaskStatus.SUCCEEDED
    assert kinds(dispatcher) == [
        (CallbackEventType.STARTED, None),
        (CallbackEventType.STAGE_COMPLETED, "extracted"),
        (CallbackEventType.STAGE_COMPLETED, "chunked"),
        (CallbackEventType.STAGE_COMPLETED, "indexed"),
        (CallbackEventType.COMPLETED, None),
    ]
    extracted = backend_callback_body(dispatcher.delivered_events[1])["metrics"]
    assert (extracted["method"], extracted["pages"], extracted["native_pages"]) == (
        "native_text",
        1,
        1,
    )
    assert extracted["characters"] > 400
    chunked = backend_callback_body(dispatcher.delivered_events[2])["metrics"]
    assert chunked["chunks"] >= 1 and chunked["pages_covered"] == 1
    indexed = backend_callback_body(dispatcher.delivered_events[3])["metrics"]
    # The backend's evidence rule: indexed_chunks equals chunked's count.
    assert indexed["indexed_chunks"] == chunked["chunks"]
    assert (indexed["index"], indexed["embedding_model"], indexed["dimension"]) == (
        "document_chunks",
        "fake-minilm",
        384,
    )
    [call] = index.calls
    assert (call["tenant_id"], call["asset_id"], call["run_number"]) == (TENANT, ASSET, 2)
    assert len(call["embeddings"]) == len(call["chunks"]) == chunked["chunks"]
    completed = backend_callback_body(dispatcher.delivered_events[-1])["metrics"]
    assert completed["chunk_summary"]["fetch"]["sha256"] == SHA
    assert transport._queue.empty()
    assert list(tmp_path.iterdir()) == []  # the fetched copy is gone


@pytest.mark.parametrize(
    "embedder,index,code",
    [
        (FakeEmbedder(EmbeddingGenerationError("onnx down", "onnx")), None, "EMBEDDING_FAILED"),
        (None, FakeIndex(VectorStoreError("db down", "pgvector")), "INDEX_WRITE_FAILED"),
    ],
    ids=["embedding", "index"],
)
async def test_an_outage_after_chunking_is_transient_and_never_indexed(
    tmp_path, embedder, index, code
):
    worker, transport, dispatcher = worker_with(
        handler(FakeS3({KEY: PDF}), tmp_path, embedder=embedder, index=index)
    )
    t = task(payload=payload())
    _, first = await run(worker, transport, t)
    assert first == TaskStatus.FAILED and not transport._queue.empty()  # retried internally

    _, final = await run(worker, transport, dataclasses.replace(t, delivery_attempt=t.max_attempts))
    failed = dispatcher.delivered_events[-1]
    assert final == TaskStatus.DEAD_LETTER
    assert (failed.error_code, failed.retryable) == (code, True)
    assert set(backend_callback_body(failed)["details"]) == {"fetch", "extraction", "chunking"}
    assert (CallbackEventType.STAGE_COMPLETED, "indexed") not in kinds(dispatcher)


async def test_a_missing_embedding_model_is_pipeline_unavailable(tmp_path, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "EMBEDDING_MODEL_DIR", None)
    h = handler(FakeS3({KEY: PDF}), tmp_path)
    h._embedder = None
    worker, transport, dispatcher = worker_with(h)

    _, status = await run(worker, transport, task(payload=payload()))

    failed = dispatcher.delivered_events[-1]
    assert status == TaskStatus.FAILED
    assert (failed.error_code, failed.retryable) == (PIPELINE_UNAVAILABLE, False)
    assert "chunking" in backend_callback_body(failed)["details"]


async def test_a_tampered_object_fails_permanently_without_evidence(tmp_path):
    worker, transport, dispatcher = worker_with(handler(FakeS3({KEY: PDF[:-1] + b"X"}), tmp_path))

    _, status = await run(worker, transport, task(payload=payload()))

    failed = dispatcher.delivered_events[-1]
    assert status == TaskStatus.FAILED
    assert (failed.error_code, failed.retryable) == ("ASSET_INTEGRITY_MISMATCH", False)
    assert "details" not in backend_callback_body(failed)
    assert transport._queue.empty()


async def test_another_tenants_reference_is_rejected_before_storage(tmp_path):
    client = FakeS3({KEY: PDF})
    other = KEY.replace(TENANT, "11111111-2222-4333-8444-555555555555")
    worker, transport, dispatcher = worker_with(handler(client, tmp_path))

    await run(worker, transport, task(payload=payload(storage_uri=f"s3://{BUCKET}/{other}")))

    assert dispatcher.delivered_events[-1].error_code == "ASSET_REFERENCE_REJECTED"
    assert client.calls == []


async def test_a_storage_outage_is_retried_internally_then_reported_retryable(tmp_path):
    client = FakeS3(error=EndpointConnectionError(endpoint_url="http://storage"))
    worker, transport, dispatcher = worker_with(handler(client, tmp_path))
    t = task(payload=payload())

    _, first = await run(worker, transport, t)
    assert first == TaskStatus.FAILED and not transport._queue.empty()  # re-queued internally

    last = dataclasses.replace(t, delivery_attempt=t.max_attempts)
    _, final = await run(worker, transport, last)
    failed = dispatcher.delivered_events[-1]
    assert final == TaskStatus.DEAD_LETTER
    assert (failed.error_code, failed.retryable) == ("OBJECT_STORAGE_UNAVAILABLE", True)


async def test_an_incomplete_dispatch_fails_permanently(tmp_path):
    worker, transport, dispatcher = worker_with(handler(FakeS3({KEY: PDF}), tmp_path))

    await run(worker, transport, task(payload={}))

    assert dispatcher.delivered_events[-1].error_code == "INVALID_INGESTION_PAYLOAD"
