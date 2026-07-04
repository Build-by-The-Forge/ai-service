from pydantic import BaseModel


class RetrieveRequest(BaseModel):
    query: str


class RetrieveResponse(BaseModel):
    success: bool
    message: str
    data: dict | None = None
