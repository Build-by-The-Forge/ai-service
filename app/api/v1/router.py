from fastapi import APIRouter

from app.api.v1.ai.exam import router as exam_router
from app.api.v1.ai.exam_submit import router as exam_submit_router
from app.api.v1.ai.fusion_exam import router as fusion_exam_router
from app.api.v1.exams import router as exams_router
from app.api.v1.health import router as health_router
from app.api.v1.system import router as system_router
from app.api.v1.ws.exam_ws import router as ws_router

api_router = APIRouter()

api_router.include_router(health_router, prefix="/health", tags=["Health"])
api_router.include_router(system_router, prefix="/system", tags=["System"])

# Legacy end-user routes (JWT + school id): mounted under /api/v1 only when
# LEGACY_USER_ROUTES_ENABLED (development only). Deleted in M4.
legacy_api_router = APIRouter()

legacy_api_router.include_router(exam_router, prefix="/ai/exam", tags=["Exam"])
legacy_api_router.include_router(exam_submit_router, prefix="/ai/exam", tags=["Exam"])
legacy_api_router.include_router(fusion_exam_router, prefix="/api/v1", tags=["AI Exam"])
legacy_api_router.include_router(exams_router, prefix="/exams", tags=["Exam Management"])
legacy_api_router.include_router(ws_router)
