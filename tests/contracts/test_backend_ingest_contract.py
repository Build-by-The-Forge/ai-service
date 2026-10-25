import json
import os

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from app.core.idempotency import InMemoryIdempotencyStore, set_idempotency_store
from app.core.replay_guard import RedisReplayGuard
from app.main import create_app
from app.schemas.ingest import (
    DispatchStatus,
    IngestDispatchRequest,
    IngestDispatchResponse,
)
from tests.signing import DISPATCH_PATH, signed_headers

REDIS_URL = os.getenv("TEST_REDIS_URL")

# The exact body BUILD backend sends from M3-06 (design B.2).
BACKEND_DISPATCH_BODY = {
    "job_id": "job_01j9a8b7c6d5e4f3a2b1c0d9e8",
    "asset_id": "asset_01j9a8b7c6d5e4f3a2b1c0d9e9",
    "source_id": "src_01j9a8b7c6d5e4f3a2b1c0d9e7",
    "run_number": 1,
    "attempt_count": 1,
    "idempotency_key": "job_01j9a8b7c6d5e4f3a2b1c0d9e8-r1-a1",
    "mime_type": "application/pdf",
    "content_sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    "file_size_bytes": 204800,
    "storage_uri": "s3://build-assets-prod/futa/cos102/2026/course_pack.pdf",
    "correlation_id": "corr_01j9a8b7c6d5e4f3a2b1c0d9ea",
}


@pytest.fixture(autouse=True)
def setup_contract_idempotency():
    set_idempotency_store(InMemoryIdempotencyStore())
    yield


def test_dispatch_contract_fields_are_exactly_the_backend_body():
    assert set(IngestDispatchRequest.model_fields) == set(BACKEND_DISPATCH_BODY)
    parsed = IngestDispatchRequest(**BACKEND_DISPATCH_BODY)
    assert parsed.run_number == 1


def test_a_backend_body_without_the_size_is_still_accepted():
    """file_size_bytes is optional until build-backend sends it (P3-1 PR 4)."""
    body = {k: v for k, v in BACKEND_DISPATCH_BODY.items() if k != "file_size_bytes"}
    assert IngestDispatchRequest(**body).file_size_bytes is None


@pytest.mark.parametrize("field", ["tenant_id", "callback_url", "user_id", "metadata"])
def test_dispatch_contract_rejects_identity_and_destination_fields(field):
    with pytest.raises(ValidationError):
        IngestDispatchRequest(**{**BACKEND_DISPATCH_BODY, field: "x"})


@pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL not set")
async def test_backend_dispatch_contract_accepted():
    """The backend's signed dispatch is accepted and answered with the agreed schema."""
    guard = RedisReplayGuard.connect(REDIS_URL, timeout=2.0)
    try:
        app = create_app(replay_guard=guard)
        body_bytes = json.dumps(BACKEND_DISPATCH_BODY).encode("utf-8")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.post(
                DISPATCH_PATH, content=body_bytes, headers=signed_headers(body_bytes)
            )
    finally:
        await guard.close()

    assert response.status_code == 202, response.text
    dispatch_response = IngestDispatchResponse(**response.json())
    assert dispatch_response.success is True
    assert dispatch_response.status == DispatchStatus.ACCEPTED
    assert dispatch_response.external_task_id.startswith("task_")
