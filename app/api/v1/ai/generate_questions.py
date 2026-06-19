from fastapi import APIRouter, Depends
from app.schemas.common import APIResponse
from app.services.ai.dependencies import get_ai_orchestrator
from app.services.ai.orchestrator import AIOrchestrator

router = APIRouter()

@router.get("/generate-questions", response_model=APIResponse)
async def generate_questions(topic: str, orchestrator: AIOrchestrator = Depends(get_ai_orchestrator)):
    result = await orchestrator.generate_questions(topic)

    return APIResponse(
        success=True,
        data=result,
        message="Questions generated",
    )
