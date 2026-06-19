from fastapi import APIRouter, Depends
from app.schemas.common import APIResponse
from app.services.ai.dependencies import get_ai_orchestrator
from app.services.ai.orchestrator import AIOrchestrator

router = APIRouter()

@router.post("/chat", response_model=APIResponse)
async def chat(question: str, orchestrator: AIOrchestrator = Depends(get_ai_orchestrator)):
    result = await orchestrator.chat(question)

    return APIResponse(
        success=True,
        data=result,
        message="Chat response generated",
    )
