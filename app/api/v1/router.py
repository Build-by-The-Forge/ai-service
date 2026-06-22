from fastapi import APIRouter
from app.api.v1.health import router as health_router
from app.api.v1.system import router as system_router
from app.api.v1.ai.chat import router as chat_router
from app.api.v1.ai.retrieve import router as retrieve_router
from app.api.v1.ai.tree import router as tree_router
from app.api.v1.ai.read import router as read_router
from app.api.v1.ai.explain import router as explain_router
from app.api.v1.ai.generate_questions import router as questions_router
from app.api.v1.ai.exam import router as exam_router
from app.api.v1.ai.exam_submit import router as exam_submit_router
from app.api.v1.ai.fusion_exam import router as fusion_exam_router
from app.api.v1.ws.exam_ws import router as ws_router
from app.api.v1.auth import router as auth_router
from app.api.v1.exams import router as exams_router

api_router = APIRouter()

api_router.include_router(health_router, prefix="/health", tags=["Health"])
api_router.include_router(system_router, prefix="/system", tags=["System"])
api_router.include_router(chat_router, prefix="/ai", tags=["AI Chat"])
api_router.include_router(retrieve_router, prefix="/ai/retrieve", tags=["RAG"])
api_router.include_router(tree_router, prefix="/ai/tree", tags=["Tree"])
api_router.include_router(read_router, prefix="/ai/read", tags=["Read Mode"])
api_router.include_router(explain_router, prefix="/ai", tags=["Explain"])
api_router.include_router(questions_router, prefix="/ai", tags=["Generation"])
api_router.include_router(exam_router, prefix="/ai/exam", tags=["Exam"])
api_router.include_router(exam_submit_router, prefix="/ai/exam", tags=["Exam"])
api_router.include_router(fusion_exam_router, prefix="/api/v1", tags=["AI Exam"])
api_router.include_router(auth_router, prefix="/auth", tags=["Auth"])
api_router.include_router(exams_router, prefix="/exams", tags=["Exam Management"])
api_router.include_router(ws_router)
