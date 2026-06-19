from fastapi import Depends, Header, HTTPException, status
from app.core.rbac import RBAC

rbac = RBAC()


def get_user_role(x_user_role: str | None = Header(None, alias="X-User-Role")) -> str:
    if not x_user_role:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-User-Role header required",
        )
    return x_user_role


def require_student(role: str = Depends(get_user_role)) -> str:
    if not rbac.can_take_exam(role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student role required",
        )
    return role


def require_teacher_or_admin(role: str = Depends(get_user_role)) -> str:
    if not rbac.can_create_exam(role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Teacher or admin role required",
        )
    return role
