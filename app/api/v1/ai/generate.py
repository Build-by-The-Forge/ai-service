from fastapi import APIRouter

router = APIRouter()

@router.post("/generate")
def generate():
    return {"success": True, "message": "generate endpoint placeholder"}
