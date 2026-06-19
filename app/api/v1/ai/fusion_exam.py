from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_db
from app.schemas.common import APIResponse
from app.services.exam.fusion_engine import CBTFusionEngine
from app.services.generation.cbt_ai_generator import CBTAIGenerator
from app.services.ai.openai import OpenAILLM
from app.services.rag.retrieval import RetrievalService
from app.services.exam.adaptive_engine import AdaptiveDifficultyEngine
from app.services.memory.memory_service import MemoryService

router = APIRouter()

@router.post("/generate", response_model=APIResponse)
async def generate_ai_exam(
    user_id: str,
    course: str,
    topic: str,
    db: AsyncSession = Depends(get_db),
):
    ai_generator = CBTAIGenerator(llm=OpenAILLM(), retriever=RetrievalService())
    adaptive_engine = AdaptiveDifficultyEngine(memory=MemoryService(db))
    engine = CBTFusionEngine(ai_generator=ai_generator, adaptive_engine=adaptive_engine)

    result = await engine.generate_personalized_exam(
        user_id=user_id,
        course=course,
        topic=topic,
    )

    return APIResponse(
        success=True,
        data=result,
        message="AI-powered exam generated",
    )
