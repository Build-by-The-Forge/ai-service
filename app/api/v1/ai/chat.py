from fastapi import APIRouter

router = APIRouter()

@router.post("/chat")
def chat():
    return {"success": True, "message": "chat endpoint placeholder"}
