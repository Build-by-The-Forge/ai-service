from fastapi import APIRouter
from app.schemas.common import APIResponse
from app.services.tree.learning_path import LearningPathService
from app.schemas.tree import TreeNode

router = APIRouter()

service = LearningPathService()

@router.get("/start", response_model=APIResponse)
async def start_reading(root: TreeNode):
    path = service.generate_path(root)

    return APIResponse(
        success=True,
        data=path,
        message="Learning path generated",
    )
