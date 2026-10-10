from fastapi import APIRouter, Depends

from app.core.service_call import signed_user_call_with
from app.core.service_signature_v1 import VerifiedCall
from app.schemas.common import APIResponse
from app.schemas.ai import ReadPathResponse, ReadStartRequest
from app.services.tree.learning_path import LearningPathService

router = APIRouter()

service = LearningPathService()


@router.post("/read/start", response_model=ReadPathResponse)
async def start_reading(
    body: ReadStartRequest,
    call: VerifiedCall = Depends(signed_user_call_with("ai.use")),
):
    path = service.generate_path(body.root)

    return APIResponse(
        success=True,
        data=path,
        message="Learning path generated",
    )
