from fastapi import APIRouter, Depends

from app.core.service_call import signed_user_call_with
from app.core.service_signature_v1 import VerifiedCall
from app.schemas.common import APIResponse
from app.schemas.ai import RetrieveRequest
from app.services.ai.tenanted_retriever import TenantAwareRetrievalService

router = APIRouter()


async def get_tenant_aware_retriever() -> TenantAwareRetrievalService:
    return TenantAwareRetrievalService()


@router.post("/retrieve", response_model=APIResponse)
async def retrieve(
    body: RetrieveRequest,
    call: VerifiedCall = Depends(signed_user_call_with("ai.use")),
    retriever: TenantAwareRetrievalService = Depends(get_tenant_aware_retriever),
):
    # Scoped to the verified tenant only (the request scope).
    results = await retriever.retrieve(body.query)

    return APIResponse(
        success=True,
        data=results,
        message="Retrieval successful",
    )
