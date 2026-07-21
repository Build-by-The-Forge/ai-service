from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.exc import IntegrityError

from app.core.dependencies import get_db
from app.auth.dependencies import (
    require_teacher_or_admin,
    require_admin,
    require_authenticated_user,
)
from app.db.models.exam import Exam
from app.schemas.common import APIResponse
from app.schemas.exam import ExamResponse

router = APIRouter()


@router.post("/", response_model=APIResponse)
async def create_exam(
    title: str,
    course: str,
    duration_minutes: int,
    is_active: bool = True,
    user: dict = Depends(require_teacher_or_admin),
    db: AsyncSession = Depends(get_db),
):
    school_id = user["school_id"]
    created_by = int(user["sub"])

    exam = Exam(
        school_id=school_id,
        title=title,
        course=course,
        duration_minutes=duration_minutes,
        is_active=is_active,
        created_by=created_by,
        updated_by=created_by,
    )
    db.add(exam)

    try:
        await db.commit()
        await db.refresh(exam)
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Exam already exists for this school",
        ) from exc

    return APIResponse(
        success=True, data=ExamResponse.from_orm(exam).dict(), message="Exam created"
    )


@router.get("/", response_model=APIResponse)
async def list_exams(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: dict = Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    school_id = user["school_id"]
    offset = (page - 1) * page_size

    result = await db.execute(
        select(Exam).where(Exam.school_id == school_id).offset(offset).limit(page_size)
    )
    exams = result.scalars().all()
    data = [ExamResponse.from_orm(exam).dict() for exam in exams]
    return APIResponse(success=True, data=data, message="Exams listed")


@router.get("/{exam_id}", response_model=APIResponse)
async def get_exam(
    exam_id: int,
    user: dict = Depends(require_authenticated_user),
    db: AsyncSession = Depends(get_db),
):
    school_id = user["school_id"]
    result = await db.execute(select(Exam).where(Exam.id == exam_id, Exam.school_id == school_id))
    exam = result.scalar_one_or_none()
    if not exam:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exam not found")
    return APIResponse(
        success=True, data=ExamResponse.from_orm(exam).dict(), message="Exam retrieved"
    )


@router.patch("/{exam_id}", response_model=APIResponse)
async def update_exam(
    exam_id: int,
    title: Optional[str] = None,
    course: Optional[str] = None,
    duration_minutes: Optional[int] = None,
    is_active: Optional[bool] = None,
    user: dict = Depends(require_teacher_or_admin),
    db: AsyncSession = Depends(get_db),
):
    school_id = user["school_id"]
    update_data: dict[str, object] = {}
    if title is not None:
        update_data["title"] = title
    if course is not None:
        update_data["course"] = course
    if duration_minutes is not None:
        update_data["duration_minutes"] = duration_minutes
    if is_active is not None:
        update_data["is_active"] = is_active

    if not update_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields to update")

    update_data["updated_by"] = int(user["sub"])
    stmt = select(Exam).where(Exam.id == exam_id, Exam.school_id == school_id)
    result = await db.execute(stmt)
    exam = result.scalar_one_or_none()
    if not exam:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exam not found")

    for key, value in update_data.items():
        setattr(exam, key, value)

    try:
        await db.commit()
        await db.refresh(exam)
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to update exam",
        ) from exc

    return APIResponse(
        success=True, data=ExamResponse.from_orm(exam).dict(), message="Exam updated"
    )


@router.delete("/{exam_id}", response_model=APIResponse)
async def delete_exam(
    exam_id: int,
    user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    school_id = user["school_id"]
    stmt = delete(Exam).where(Exam.id == exam_id, Exam.school_id == school_id)
    result = await db.execute(stmt)
    if result.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Exam not found")
    await db.commit()
    return APIResponse(success=True, message="Exam deleted")
