from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_db
from app.schemas.common import APIResponse
from app.services.exam.scoring import ScoringEngine

router = APIRouter()
scoring = ScoringEngine()

@router.post("/submit", response_model=APIResponse)
async def submit_answer(
    correct_answer: str,
    user_answer: str,
    db: AsyncSession = Depends(get_db),
):
    is_correct = scoring.grade(correct_answer, user_answer)

    return APIResponse(
        success=True,
        data={"is_correct": is_correct},
        message="Answer graded",
    )
