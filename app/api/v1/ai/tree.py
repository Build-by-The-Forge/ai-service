from fastapi import APIRouter, Depends

from app.core.service_call import signed_user_call_with
from app.core.service_signature_v1 import VerifiedCall
from app.schemas.common import APIResponse
from app.schemas.ai import TreeGenerateRequest, TreeResponse
from app.services.tree.generator import TreeGeneratorService

router = APIRouter()

generator = TreeGeneratorService()


@router.post("/tree/generate", response_model=TreeResponse)
async def generate_tree(
    body: TreeGenerateRequest,
    call: VerifiedCall = Depends(signed_user_call_with("ai.use")),
):
    tree = generator.generate_base_tree(body.course_title, body.topics)

    return APIResponse(
        success=True,
        data=tree,
        message="Tree generated",
    )
