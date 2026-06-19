from fastapi import APIRouter

router = APIRouter()

@router.post("/tree")
def tree():
    return {"success": True, "message": "tree endpoint placeholder"}
