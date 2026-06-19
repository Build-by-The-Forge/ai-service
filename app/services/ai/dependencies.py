from app.services.ai.openai import OpenAILLM
from app.services.ai.orchestrator import AIOrchestrator
from app.services.rag.retrieval import RetrievalService


def get_ai_orchestrator() -> AIOrchestrator:
    return AIOrchestrator(llm=OpenAILLM(), retriever=RetrievalService())
