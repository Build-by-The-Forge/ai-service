from fastapi import APIRouter, Depends
from app.schemas.common import APIResponse
from app.services.ai.tenanted_retriever import TenantAwareRetrievalService

router = APIRouter()

@router.get("/", response_model=APIResponse)
async def retrieve(query: str, retriever: TenantAwareRetrievalService = Depends()):
    results = await retriever.retrieve(query)

    return APIResponse(
        success=True,
        data=results,
        message="Retrieval successful",
    )
