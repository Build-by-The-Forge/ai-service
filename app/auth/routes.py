from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from passlib.hash import pbkdf2_sha256

from app.core.dependencies import get_db
from app.db.models.user import User
from app.auth.token import create_jwt
from app.auth.dependencies import require_authenticated_user
from app.schemas.common import APIResponse

router = APIRouter()


@router.post("/login", response_model=APIResponse)
async def login(email: str, password: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user or not pbkdf2_sha256.verify(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    token = create_jwt(subject=str(user.id), school_id=user.school_id, role=user.role)
    return APIResponse(
        success=True, data={"access_token": token, "token_type": "bearer"}, message="Logged in"
    )


@router.post("/refresh", response_model=APIResponse)
async def refresh(user: dict = Depends(require_authenticated_user)):
    token = create_jwt(subject=str(user["sub"]), school_id=user["school_id"], role=user["role"])
    return APIResponse(
        success=True,
        data={"access_token": token, "token_type": "bearer"},
        message="Token refreshed",
    )


@router.get("/me", response_model=APIResponse)
async def me(user: dict = Depends(require_authenticated_user)):
    return APIResponse(success=True, data=user, message="User profile")


@router.post("/logout", response_model=APIResponse)
async def logout():
    return APIResponse(success=True, message="Logged out")
