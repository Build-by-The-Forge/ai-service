from fastapi import APIRouter, Depends

from app.core.service_call import signed_user_call_with
from app.core.service_signature_v1 import VerifiedCall
from app.schemas.common import APIResponse
from app.schemas.ai import ChatRequest, ChatResponse
from app.services.ai.dependencies import get_ai_orchestrator
from app.services.ai.orchestrator import AIOrchestrator

router = APIRouter()

AI_USE_PERMISSION = "ai.use"


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    call: VerifiedCall = Depends(signed_user_call_with(AI_USE_PERMISSION)),
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator),
):
    # Memory is the verified user's, in the verified tenant (design B.4).
    user_id = call.context.user_id or ""
    result = await orchestrator.chat(user_id=user_id, question=body.message)

    return APIResponse(
        success=True,
        data=result,
        message="Chat response generated",
    )
