from fastapi import APIRouter
from app.api.v1.ai import chat, read, tree, retrieve, generate, ocr

ai_router = APIRouter()

ai_router.include_router(chat.router, prefix="/chat", tags=["AI"])
ai_router.include_router(read.router, prefix="/read", tags=["AI"])
ai_router.include_router(tree.router, prefix="/tree", tags=["AI"])
ai_router.include_router(retrieve.router, prefix="/retrieve", tags=["AI"])
ai_router.include_router(generate.router, prefix="/generate", tags=["AI"])
ai_router.include_router(ocr.router, prefix="/ocr", tags=["AI"])
