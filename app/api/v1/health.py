from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def health_check():
    return {
        "success": True,
        "data": {
            "status": "healthy",
            "service": "ai-service"
        },
        "message": "OK"
    }
