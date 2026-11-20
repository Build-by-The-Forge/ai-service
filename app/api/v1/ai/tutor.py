"""One AI Tutor conversation turn (M3-F, docs/m3-f-tutor-turn.md).

The backend owns the conversation and stores the turn once its provenance
check passes; nothing of it is kept here. The response carries the answer,
the passages it cites as structured citations, and every source retrieval
returned, cited or not. Uncited passages and memory are never returned.
"""

from fastapi import APIRouter, Depends

from app.core.service_call import signed_user_call_with
from app.core.service_signature_v1 import VerifiedCall
from app.core.signed_response import signed_response
from app.schemas.ai import TutorTurnRequest, TutorTurnResponse
from app.services.ai.dependencies import get_ai_orchestrator
from app.services.ai.gateway_llm import invocation_context
from app.services.ai.orchestrator import AIOrchestrator

router = APIRouter()

AI_USE_PERMISSION = "ai.use"


@router.post("/tutor/turn", response_model=TutorTurnResponse)
async def tutor_turn(
    body: TutorTurnRequest,
    call: VerifiedCall = Depends(signed_user_call_with(AI_USE_PERMISSION)),
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator),
):
    result = await orchestrator.tutor_turn(
        question=body.message,
        earlier_questions=list(body.earlier_questions),
        denied_assets=frozenset(body.denied_assets),
        invocation=invocation_context(call, "chat"),
    )
    sources_used = result.pop("sources_used")
    payload = TutorTurnResponse(
        data=result, message="Tutor turn answered", sources_used=sources_used
    )
    return signed_response(call, body.response_nonce, payload)
