from datetime import datetime, timedelta
from typing import Any

from jose import JWTError, jwt
from app.core.config import settings


def create_jwt(subject: str, school_id: int, role: str, expires_minutes: int = 60) -> str:
    now = datetime.utcnow()
    payload: dict[str, Any] = {
        "sub": subject,
        "school_id": school_id,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
        "aud": settings.JWT_AUDIENCE,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_jwt(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
            audience=settings.JWT_AUDIENCE,
        )
        return payload
    except JWTError as exc:
        raise ValueError("Invalid JWT token") from exc
