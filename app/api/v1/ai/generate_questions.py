from fastapi import APIRouter, Depends

from app.core.service_call import signed_user_call_with
from app.core.service_signature_v1 import VerifiedCall
from app.schemas.common import APIResponse
from app.schemas.ai import GeneratedQuestionsResponse, GenerateQuestionsRequest
from app.services.ai.dependencies import get_ai_orchestrator
from app.services.ai.orchestrator import AIOrchestrator

router = APIRouter()


@router.post("/generate-questions", response_model=GeneratedQuestionsResponse)
async def generate_questions(
    body: GenerateQuestionsRequest,
    call: VerifiedCall = Depends(signed_user_call_with("ai.questions.generate")),
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator),
):
    result = await orchestrator.generate_questions(
        body.topic,
        count=body.count,
        difficulty=body.difficulty,
        question_type=body.question_type,
    )

    return APIResponse(
        success=True,
        data=result,
        message="Questions generated",
    )
