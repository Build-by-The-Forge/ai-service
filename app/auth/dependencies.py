from fastapi import Depends, Header, HTTPException, status
from app.auth.permissions import rbac
from app.auth.token import decode_jwt
from app.core.tenant import set_school_id


def get_authorization_header(
    authorization: str | None = Header(None, alias="Authorization")
) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing or malformed",
        )
    return authorization.split(" ", 1)[1]


async def require_authenticated_user(token: str = Depends(get_authorization_header)) -> dict:
    try:
        payload = decode_jwt(token)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )

    if not payload.get("sub") or not payload.get("school_id") or not payload.get("role"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )

    set_school_id(int(payload["school_id"]))
    return payload


async def require_student(user: dict = Depends(require_authenticated_user)) -> dict:
    if not rbac.can_take_exam(user["role"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student role required",
        )
    return user


async def require_teacher_or_admin(user: dict = Depends(require_authenticated_user)) -> dict:
    if not rbac.can_create_exam(user["role"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Teacher or admin role required",
        )
    return user


async def require_admin(user: dict = Depends(require_authenticated_user)) -> dict:
    if not rbac.can_manage_users(user["role"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return user
