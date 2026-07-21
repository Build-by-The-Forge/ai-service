from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from app.core.config import settings


def create_jwt(subject: str, school_id: int, role: str, expires_minutes: int = 60) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "school_id": school_id,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
    }
    if settings.JWT_AUDIENCE:
        payload["aud"] = settings.JWT_AUDIENCE
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_jwt(token: str) -> dict[str, Any]:
    try:
        decode_kwargs: dict[str, Any] = {
            "key": settings.JWT_SECRET,
            "algorithms": [settings.JWT_ALGORITHM],
        }
        if settings.JWT_AUDIENCE:
            decode_kwargs["audience"] = settings.JWT_AUDIENCE
        payload = jwt.decode(token, **decode_kwargs)
        return payload
    except JWTError as exc:
        raise ValueError("Invalid JWT token") from exc
