from fastapi import APIRouter

router = APIRouter()

@router.post("/ocr")
def ocr():
    return {"success": True, "message": "ocr endpoint placeholder"}
