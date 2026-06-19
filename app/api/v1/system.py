from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health():
    return {
        "success": True,
        "data": {"status": "ok"},
        "message": "healthy",
    }

@router.get("/ready")
async def readiness():
    return {
        "success": True,
        "data": {"ready": True},
        "message": "ready",
    }
