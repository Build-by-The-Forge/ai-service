from typing import Any

from fastapi import HTTPException, status

from app.core.config import settings


class TokenVerifier:
    """JWT token verification (integration with external auth service)."""

    async def verify(self, token: str) -> dict[str, Any]:
        """
        Verify a JWT token.
        In production, this integrates with your JWKS endpoint or auth service.
        """
        if not settings.JWT_PUBLIC_KEY:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="JWT public key not configured",
            )

        try:
            # Placeholder for real JWT verification
            # In production: use python-jose or cryptography to validate RS256
            payload = {
                "sub": "user_id",
                "school_id": 1,
                "role": "student",
            }
            return payload

        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
            ) from exc
