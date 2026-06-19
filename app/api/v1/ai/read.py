from fastapi import APIRouter

router = APIRouter()

@router.post("/read")
def read():
    return {"success": True, "message": "read endpoint placeholder"}
