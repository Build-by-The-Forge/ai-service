from pydantic import BaseModel

class BaseResponse(BaseModel):
    success: bool
    message: str
    data: dict | None = None
