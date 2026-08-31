from fastapi import APIRouter, Depends

from app.schemas.common import APIResponse
from app.services.ai.dependencies import get_ai_orchestrator
from app.services.ai.orchestrator import AIOrchestrator

router = APIRouter()


@router.post("/chat", response_model=APIResponse)
async def chat(
    user_id: str,
    question: str,
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator),
):
    result = await orchestrator.chat(user_id=user_id, question=question)

    return APIResponse(
        success=True,
        data=result,
        message="Chat response generated",
    )
