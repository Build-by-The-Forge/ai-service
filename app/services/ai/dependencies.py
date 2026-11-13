from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models.gateway import get_model_gateway
from app.db.session import get_async_session_local
from app.services.ai.gateway_llm import GatewayLLM
from app.services.ai.orchestrator import AIOrchestrator
from app.services.ai.tenanted_retriever import TenantAwareRetrievalService
from app.services.memory.memory_service import MemoryService


async def get_db_session():
    session_factory = get_async_session_local()
    async with session_factory() as session:
        yield session


async def get_ai_orchestrator(db: AsyncSession = Depends(get_db_session)) -> AIOrchestrator:
    # Chat, explain and question generation run through the one model
    # execution boundary (AI-01 PR 3). In deterministic mode the default
    # gateway is offline: its adapter is the only one registered or approved.
    return AIOrchestrator(
        llm=GatewayLLM(get_model_gateway()),
        retriever=TenantAwareRetrievalService(),
        memory=MemoryService(db),
    )
