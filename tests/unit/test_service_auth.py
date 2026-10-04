import time

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.core.service_auth import (
    AuthenticatedServiceContext,
    compute_service_signature,
    verify_service_authentication,
)


@pytest.fixture
def auth_test_app():
    app = FastAPI()

    @app.post("/test-service-auth")
    async def protected_endpoint(
        ctx: AuthenticatedServiceContext = Depends(verify_service_authentication),
    ):
        return {"status": "ok", "service_identity": ctx.service_identity}

    return app


@pytest.mark.asyncio
async def test_valid_service_authentication_passes(auth_test_app):
    transport = ASGITransport(app=auth_test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        timestamp = int(time.time())
        body = b'{"action": "test"}'
        sig = compute_service_signature(
            method="POST",
            path="/test-service-auth",
            timestamp=timestamp,
            body=body,
            secret=settings.BACKEND_TO_AI_SECRET,
        )

        response = await client.post(
            "/test-service-auth",
            content=body,
            headers={
                "X-Service-Identity": "build-backend",
                "X-Timestamp": str(timestamp),
                "X-Service-Signature": sig,
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 200
        assert response.json()["service_identity"] == "build-backend"


@pytest.mark.asyncio
async def test_missing_auth_headers_returns_401(auth_test_app):
    transport = ASGITransport(app=auth_test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/test-service-auth", json={"action": "test"})
        assert response.status_code == 401
        assert "Missing required service authentication headers" in response.json()["detail"]


@pytest.mark.asyncio
async def test_unauthorized_service_identity_returns_403(auth_test_app):
    transport = ASGITransport(app=auth_test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        timestamp = int(time.time())
        body = b'{"action": "test"}'
        sig = compute_service_signature(
            method="POST",
            path="/test-service-auth",
            timestamp=timestamp,
            body=body,
            secret=settings.BACKEND_TO_AI_SECRET,
        )

        response = await client.post(
            "/test-service-auth",
            content=body,
            headers={
                "X-Service-Identity": "unknown-service",
                "X-Timestamp": str(timestamp),
                "X-Service-Signature": sig,
            },
        )
        assert response.status_code == 403
        assert "not authorized" in response.json()["detail"]


@pytest.mark.asyncio
async def test_expired_timestamp_skew_returns_401(auth_test_app):
    transport = ASGITransport(app=auth_test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Timestamp 10 minutes in the past (> 300s window)
        stale_timestamp = int(time.time()) - 600
        body = b'{"action": "test"}'
        sig = compute_service_signature(
            method="POST",
            path="/test-service-auth",
            timestamp=stale_timestamp,
            body=body,
            secret=settings.BACKEND_TO_AI_SECRET,
        )

        response = await client.post(
            "/test-service-auth",
            content=body,
            headers={
                "X-Service-Identity": "build-backend",
                "X-Timestamp": str(stale_timestamp),
                "X-Service-Signature": sig,
            },
        )
        assert response.status_code == 401
        assert "Timestamp skew exceeded" in response.json()["detail"]


@pytest.mark.asyncio
async def test_tampered_body_signature_mismatch_returns_401(auth_test_app):
    transport = ASGITransport(app=auth_test_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        timestamp = int(time.time())
        original_body = b'{"action": "test"}'
        sig = compute_service_signature(
            method="POST",
            path="/test-service-auth",
            timestamp=timestamp,
            body=original_body,
            secret=settings.BACKEND_TO_AI_SECRET,
        )

        tampered_body = b'{"action": "tampered"}'
        response = await client.post(
            "/test-service-auth",
            content=tampered_body,
            headers={
                "X-Service-Identity": "build-backend",
                "X-Timestamp": str(timestamp),
                "X-Service-Signature": sig,
            },
        )
        assert response.status_code == 401
        assert "Invalid service authentication signature" in response.json()["detail"]
