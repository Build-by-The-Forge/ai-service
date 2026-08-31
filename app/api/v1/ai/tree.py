from fastapi import APIRouter

from app.schemas.common import APIResponse
from app.services.tree.generator import TreeGeneratorService

router = APIRouter()

generator = TreeGeneratorService()


@router.post("/generate", response_model=APIResponse)
async def generate_tree(course_title: str, topics: list[str]):
    tree = generator.generate_base_tree(course_title, topics)

    return APIResponse(
        success=True,
        data=tree,
        message="Tree generated",
    )
