from fastapi import APIRouter, Depends

from app.core.signed_response import signed_response
from app.core.service_call import signed_user_call_with
from app.core.service_signature_v1 import VerifiedCall
from app.schemas.ai import ExplainRequest, ExplainResponse
from app.services.ai.dependencies import get_ai_orchestrator
from app.services.ai.gateway_llm import invocation_context
from app.services.ai.orchestrator import AIOrchestrator

router = APIRouter()


@router.post("/explain", response_model=ExplainResponse)
async def explain(
    body: ExplainRequest,
    call: VerifiedCall = Depends(signed_user_call_with("ai.use")),
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator),
):
    result = await orchestrator.explain(
        body.subject,
        frozenset(body.denied_assets),
        invocation=invocation_context(call, "explain"),
    )

    sources_used = result.pop("sources_used")
    payload = ExplainResponse(
        data=result, message="Explanation generated", sources_used=sources_used
    )
    return signed_response(call, body.response_nonce, payload)
