from fastapi import APIRouter, Depends
from app.schemas.common import APIResponse
from app.services.rag.retrieval import RetrievalService

router = APIRouter()

@router.get("/", response_model=APIResponse)
async def retrieve(query: str, retriever: RetrievalService = Depends()):
    results = await retriever.retrieve(query)

    return APIResponse(
        success=True,
        data=results,
        message="Retrieval successful",
    )
