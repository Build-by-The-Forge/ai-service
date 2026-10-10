from fastapi import APIRouter, Depends

from app.core.service_call import signed_user_call_with
from app.core.service_signature_v1 import VerifiedCall
from app.schemas.common import APIResponse
from app.schemas.ai import ExplainRequest
from app.services.ai.dependencies import get_ai_orchestrator
from app.services.ai.orchestrator import AIOrchestrator

router = APIRouter()


@router.post("/explain", response_model=APIResponse)
async def explain(
    body: ExplainRequest,
    call: VerifiedCall = Depends(signed_user_call_with("ai.use")),
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator),
):
    result = await orchestrator.explain(body.subject)

    return APIResponse(
        success=True,
        data=result,
        message="Explanation generated",
    )
