import json
import time
import pytest
from httpx import AsyncClient, ASGITransport

from app.core.config import settings
from app.core.idempotency import InMemoryIdempotencyStore, set_idempotency_store
from app.core.service_auth import compute_service_signature
from app.main import app


@pytest.fixture(autouse=True)
def setup_test_idempotency():
    store = InMemoryIdempotencyStore()
    set_idempotency_store(store)
    yield


@pytest.mark.asyncio
async def test_dispatch_ingest_task_accepted():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        payload = {
            "job_id": "job_01h8",
            "asset_id": "asset_99a",
            "tenant_id": "1",
            "source": "upload",
            "authority": "INSTITUTIONAL",
            "storage_uri": "s3://build-assets/courses/cos102/syllabus.pdf",
            "mime_type": "application/pdf",
            "content_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "run_number": 1,
            "attempt_count": 1,
            "correlation_id": "corr_abc_123",
            "idempotency_key": "idemp_unique_001",
            "callback_url": "https://backend.build.edu/api/v1/ingest/callback",
            "metadata": {"course": "COS 102", "level": 100},
        }
        body_bytes = json.dumps(payload).encode("utf-8")
        timestamp = int(time.time())
        signature = compute_service_signature(
            method="POST",
            path="/v1/ingest/dispatch",
            timestamp=timestamp,
            body=body_bytes,
            secret=settings.SERVICE_AUTH_SECRET,
        )

        response = await client.post(
            "/v1/ingest/dispatch",
            content=body_bytes,
            headers={
                "X-Service-Identity": "build-backend",
                "X-Timestamp": str(timestamp),
                "X-Service-Signature": signature,
                "Content-Type": "application/json",
            },
        )

        assert response.status_code == 202
        body = response.json()
        assert body["success"] is True
        assert body["status"] == "ACCEPTED"
        assert body["external_task_id"].startswith("task_")
        assert body["data"]["job_id"] == "job_01h8"
        assert body["data"]["run_number"] == 1


@pytest.mark.asyncio
async def test_dispatch_duplicate_request_returns_duplicate_status():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        payload = {
            "job_id": "job_01h8",
            "asset_id": "asset_99a",
            "tenant_id": "1",
            "source": "upload",
            "authority": "INSTITUTIONAL",
            "storage_uri": "s3://build-assets/courses/cos102/syllabus.pdf",
            "mime_type": "application/pdf",
            "content_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "run_number": 1,
            "attempt_count": 1,
            "correlation_id": "corr_abc_123",
            "idempotency_key": "idemp_duplicate_002",
            "callback_url": "https://backend.build.edu/api/v1/ingest/callback",
        }
        body_bytes = json.dumps(payload).encode("utf-8")
        timestamp = int(time.time())
        signature = compute_service_signature(
            method="POST",
            path="/v1/ingest/dispatch",
            timestamp=timestamp,
            body=body_bytes,
            secret=settings.SERVICE_AUTH_SECRET,
        )

        # First request
        resp1 = await client.post(
            "/v1/ingest/dispatch",
            content=body_bytes,
            headers={
                "X-Service-Identity": "build-backend",
                "X-Timestamp": str(timestamp),
                "X-Service-Signature": signature,
                "Content-Type": "application/json",
            },
        )
        assert resp1.status_code == 202
        first_task_id = resp1.json()["external_task_id"]

        # Duplicate request (same idempotency_key and tenant_id)
        resp2 = await client.post(
            "/v1/ingest/dispatch",
            content=body_bytes,
            headers={
                "X-Service-Identity": "build-backend",
                "X-Timestamp": str(timestamp),
                "X-Service-Signature": signature,
                "Content-Type": "application/json",
            },
        )
        assert resp2.status_code == 202
        body2 = resp2.json()
        assert body2["success"] is True
        assert body2["status"] == "DUPLICATE"
        assert body2["external_task_id"] == first_task_id


@pytest.mark.asyncio
async def test_dispatch_invalid_sha256_returns_422():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        invalid_payload = {
            "job_id": "job_01h8",
            "asset_id": "asset_99a",
            "tenant_id": "1",
            "source": "upload",
            "authority": "INSTITUTIONAL",
            "storage_uri": "s3://build-assets/courses/cos102/syllabus.pdf",
            "mime_type": "application/pdf",
            "content_sha256": "invalid_short_hash",
            "run_number": 1,
            "attempt_count": 1,
            "correlation_id": "corr_abc_123",
            "idempotency_key": "idemp_err_003",
            "callback_url": "https://backend.build.edu/api/v1/ingest/callback",
        }
        body_bytes = json.dumps(invalid_payload).encode("utf-8")
        timestamp = int(time.time())
        signature = compute_service_signature(
            method="POST",
            path="/v1/ingest/dispatch",
            timestamp=timestamp,
            body=body_bytes,
            secret=settings.SERVICE_AUTH_SECRET,
        )

        response = await client.post(
            "/v1/ingest/dispatch",
            content=body_bytes,
            headers={
                "X-Service-Identity": "build-backend",
                "X-Timestamp": str(timestamp),
                "X-Service-Signature": signature,
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 422
