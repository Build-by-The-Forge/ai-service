"""Helpers for tests that call signed AI-service routes as the backend would."""

import json
import time
from typing import Any

from app.core.config import settings
from app.core.service_context import ServiceContext, parse_service_context
from app.core.service_signature_v1 import BACKEND_SERVICE_ID, ServiceSigner

TENANT_A = "6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c"
TENANT_B = "9d8c7b6a-5f4e-4d3c-8b2a-1f0e9d8c7b6a"
USER_ID = "0b9d8c7a-6e5f-4a3b-8c2d-1e0f9a8b7c6d"
MEMBERSHIP_ID = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
DISPATCH_PATH = "/v1/ingest/dispatch"


def user_context(
    tenant_id: str = TENANT_A,
    permissions: tuple[str, ...] = ("content.job.manage",),
    role: str = "admin",
) -> ServiceContext:
    return parse_service_context(
        {
            "v": 1,
            "actor": "user",
            "tenant_id": tenant_id,
            "user_id": USER_ID,
            "membership_id": MEMBERSHIP_ID,
            "role": role,
            "permissions": sorted(permissions),
            "correlation_id": "corr-test",
        }
    )


def dispatch_body(**overrides: Any) -> bytes:
    body = {
        "job_id": "job-1",
        "asset_id": "asset-1",
        "source_id": "source-1",
        "run_number": 1,
        "attempt_count": 1,
        "idempotency_key": "job-1-r1-a1",
        "mime_type": "application/pdf",
        "content_sha256": "ab" * 32,
        "storage_uri": "s3://build-assets/tenant-a/syllabus.pdf",
        "correlation_id": "corr-test",
    }
    body.update(overrides)
    return json.dumps({k: v for k, v in body.items() if v is not None}).encode()


def signed_headers(
    body: bytes,
    *,
    context: ServiceContext | None = None,
    target: str = DISPATCH_PATH,
    method: str = "POST",
    secret: str | None = None,
    sender: str = BACKEND_SERVICE_ID,
    timestamp: int | None = None,
) -> dict[str, str]:
    signer = ServiceSigner(sender=sender, secret=secret or settings.BACKEND_TO_AI_SECRET or "")
    headers = signer.headers(
        method=method,
        target=target,
        body=body,
        context=context or user_context(),
        timestamp=int(time.time()) if timestamp is None else timestamp,
    )
    headers["Content-Type"] = "application/json"
    return headers
