"""The ingestion handler in the worker lifecycle (P3-1 PR 5): the asset is
fetched and verified first; the task then still fails honestly with
INGESTION_PIPELINE_UNAVAILABLE, carrying the fetch evidence to the backend;
a failed fetch reports its own typed failure, retried only if transient."""

import dataclasses
import hashlib

from botocore.exceptions import EndpointConnectionError

from app.core.callbacks.dispatcher import backend_callback_body
from app.core.callbacks.events import CallbackEventType
from app.core.execution_context import TaskStatus
from app.services.ingestion.asset_fetcher import AssetFetcher
from app.services.tasks.handlers import ContentIngestionTaskHandler
from app.services.tasks.ingestion_outcomes import PIPELINE_UNAVAILABLE
from tests.unit.test_asset_fetcher import FakeS3
from tests.unit.test_worker_callback_integration import kinds, run, task, worker_with

BUCKET = "build-documents"
TENANT = "6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c"  # the task's verified tenant
ASSET = "0b9a8c7d-6e5f-5a4b-8c3d-2e1f0a9b8c7d"
PDF = b"%PDF-1.7\n" + b"lecture notes " * 500
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


def handler(client, tmp_path) -> ContentIngestionTaskHandler:
    return ContentIngestionTaskHandler(
        AssetFetcher(
            bucket=BUCKET,
            client_factory=lambda: client,
            max_bytes=10 * 1024 * 1024,
            timeout_seconds=5,
            temp_dir=str(tmp_path),
        )
    )


async def test_a_verified_fetch_is_reported_as_evidence_then_the_pipeline_is_unavailable(tmp_path):
    worker, transport, dispatcher = worker_with(handler(FakeS3({KEY: PDF}), tmp_path))

    msg_id, status = await run(worker, transport, task(payload=payload()))

    assert status == TaskStatus.FAILED
    assert kinds(dispatcher) == [
        (CallbackEventType.STARTED, None),
        (CallbackEventType.FAILED, None),
    ]
    failed = dispatcher.delivered_events[-1]
    assert (failed.error_code, failed.retryable) == (PIPELINE_UNAVAILABLE, False)
    body = backend_callback_body(failed)
    assert body["details"]["fetch"] | {"seconds": 0} == {
        "verified": True,
        "size_bytes": len(PDF),
        "sha256": SHA,
        "mime_type": "application/pdf",
        "seconds": 0,
    }
    assert transport._queue.empty()  # not retried internally
    assert list(tmp_path.iterdir()) == []  # the fetched copy is gone


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
