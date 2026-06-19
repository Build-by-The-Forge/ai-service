from pydantic import BaseModel

class TreeRequest(BaseModel):
    text: str

class TreeResponse(BaseModel):
    success: bool
    message: str
    data: dict | None = None
