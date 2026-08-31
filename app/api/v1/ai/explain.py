from fastapi import APIRouter, Depends

from app.schemas.common import APIResponse
from app.services.ai.dependencies import get_ai_orchestrator
from app.services.ai.orchestrator import AIOrchestrator

router = APIRouter()


@router.get("/explain", response_model=APIResponse)
async def explain(subject: str, orchestrator: AIOrchestrator = Depends(get_ai_orchestrator)):
    result = await orchestrator.explain(subject)

    return APIResponse(
        success=True,
        data=result,
        message="Explanation generated",
    )
