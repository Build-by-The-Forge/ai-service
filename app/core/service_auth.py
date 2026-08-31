import hashlib
import hmac
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from app.core.config import settings


@dataclass(frozen=True)
class AuthenticatedServiceContext:
    service_identity: str
    timestamp: int
    is_authenticated: bool = True


def compute_service_signature(
    method: str,
    path: str,
    timestamp: int,
    body: bytes | str,
    secret: str,
) -> str:
    """
    Computes HMAC-SHA256 signature for service-to-service communication.
    Canonical format: {METHOD}\n{PATH}\n{TIMESTAMP}\n{BODY}
    """
    if isinstance(body, str):
        body_bytes = body.encode("utf-8")
    else:
        body_bytes = body

    canonical_payload = f"{method.upper()}\n{path}\n{timestamp}\n".encode() + body_bytes
    return hmac.new(secret.encode("utf-8"), canonical_payload, hashlib.sha256).hexdigest()


async def verify_service_authentication(request: Request) -> AuthenticatedServiceContext:
    """
    FastAPI dependency that validates service identity, timestamp freshness, and HMAC signature.
    """
    service_id = request.headers.get("X-Service-Identity")
    timestamp_header = request.headers.get("X-Timestamp")
    signature = request.headers.get("X-Service-Signature")

    if not service_id or not timestamp_header or not signature:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required service authentication headers (X-Service-Identity, X-Timestamp, X-Service-Signature)",
        )

    # 1. Check allowed service identities
    if service_id not in settings.ALLOWED_SERVICE_IDENTITIES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Service identity '{service_id}' is not authorized",
        )

    # 2. Validate timestamp freshness (replay attack prevention)
    try:
        req_timestamp = int(timestamp_header)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid X-Timestamp header format; integer epoch seconds required",
        )

    current_time = int(time.time())
    if abs(current_time - req_timestamp) > settings.SERVICE_AUTH_MAX_SKEW_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Timestamp skew exceeded acceptable window",
        )

    # 3. Verify HMAC signature over request body
    body_bytes = await request.body()
    expected_signature = compute_service_signature(
        method=request.method,
        path=request.url.path,
        timestamp=req_timestamp,
        body=body_bytes,
        secret=settings.SERVICE_AUTH_SECRET,
    )

    if not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service authentication signature",
        )

    return AuthenticatedServiceContext(
        service_identity=service_id,
        timestamp=req_timestamp,
        is_authenticated=True,
    )
