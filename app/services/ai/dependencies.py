from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.ai.openai import OpenAILLM
from app.services.ai.orchestrator import AIOrchestrator
from app.services.ai.tenanted_retriever import TenantAwareRetrievalService
from app.services.memory.memory_service import MemoryService
from app.db.session import get_async_session_local


async def get_db_session():
    session_factory = get_async_session_local()
    async with session_factory() as session:
        yield session


async def get_ai_orchestrator(db: AsyncSession = Depends(get_db_session)) -> AIOrchestrator:
    return AIOrchestrator(
        llm=OpenAILLM(),
        retriever=TenantAwareRetrievalService(),
        memory=MemoryService(db),
    )
