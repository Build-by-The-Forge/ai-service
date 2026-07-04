from pydantic import BaseModel


class OCRRequest(BaseModel):
    file_url: str


class OCRResponse(BaseModel):
    success: bool
    message: str
    data: dict | None = None
