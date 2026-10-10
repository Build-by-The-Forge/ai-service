"""Signed AI routes, mounted at /v1/ai (design B.4).

Reachable only with a verified BUILD Service Signature v1 request carrying a
user context with the route's permission. The backend proxies chat, explain
and generate-questions; retrieve, tree and read have no client path until M5.
"""

from fastapi import APIRouter

from app.api.v1.ai.chat import router as chat_router
from app.api.v1.ai.explain import router as explain_router
from app.api.v1.ai.generate_questions import router as questions_router
from app.api.v1.ai.read import router as read_router
from app.api.v1.ai.retrieve import router as retrieve_router
from app.api.v1.ai.tree import router as tree_router

ai_router = APIRouter()

ai_router.include_router(chat_router, tags=["AI Chat"])
ai_router.include_router(explain_router, tags=["Explain"])
ai_router.include_router(questions_router, tags=["Generation"])
ai_router.include_router(retrieve_router, tags=["RAG"])
ai_router.include_router(tree_router, tags=["Tree"])
ai_router.include_router(read_router, tags=["Read Mode"])
