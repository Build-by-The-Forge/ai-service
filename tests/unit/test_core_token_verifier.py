import pytest
from fastapi import HTTPException, status

from app.core.config import settings
from app.core.token_verifier import TokenVerifier


@pytest.mark.asyncio
async def test_token_verifier_raises_when_public_key_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "JWT_PUBLIC_KEY", None)

    verifier = TokenVerifier()

    with pytest.raises(HTTPException) as exc_info:
        await verifier.verify("invalid-token")

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "JWT public key not configured" in exc_info.value.detail


@pytest.mark.asyncio
async def test_token_verifier_returns_payload_when_public_key_is_configured(monkeypatch):
    monkeypatch.setattr(settings, "JWT_PUBLIC_KEY", "fake-public-key")

    verifier = TokenVerifier()
    payload = await verifier.verify("any-token")

    assert payload["sub"] == "user_id"
    assert payload["school_id"] == 1
    assert payload["role"] == "student"
