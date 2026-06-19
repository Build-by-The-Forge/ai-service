from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_db
from app.schemas.common import APIResponse
from app.services.exam.exam_engine import ExamEngine

router = APIRouter()

@router.post("/start", response_model=APIResponse)
async def start_exam(exam_id: int, user_id: str, db: AsyncSession = Depends(get_db)):
    engine = ExamEngine(db)
    result = await engine.start_exam(exam_id, user_id)

    return APIResponse(
        success=True,
        data=result,
        message="Exam started",
    )
