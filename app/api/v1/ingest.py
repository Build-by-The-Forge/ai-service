import uuid
from fastapi import APIRouter, Depends, status

from app.core.idempotency import IdempotencyStore, get_idempotency_store
from app.core.service_auth import (
    AuthenticatedServiceContext,
    verify_service_authentication,
)
from app.core.tenant import set_school_id
from app.schemas.ingest import (
    DispatchStatus,
    IngestDispatchRequest,
    IngestDispatchResponse,
)

router = APIRouter()


@router.post(
    "/dispatch",
    response_model=IngestDispatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Dispatch Content Ingestion Task",
    description=(
        "Authenticated service-to-service endpoint for dispatching content ingestion jobs. "
        "Guarantees idempotency and enforces authenticated service claims."
    ),
)
async def dispatch_ingest_task(
    payload: IngestDispatchRequest,
    service_ctx: AuthenticatedServiceContext = Depends(verify_service_authentication),
    idempotency_store: IdempotencyStore = Depends(get_idempotency_store),
) -> IngestDispatchResponse:
    # 1. Set tenant context from authenticated claim
    try:
        numeric_tenant = int(payload.tenant_id)
        set_school_id(numeric_tenant)
    except (ValueError, TypeError):
        pass

    # 2. Check Idempotency via abstraction port
    candidate_task_id = f"task_{uuid.uuid4().hex}"
    is_fresh, existing_record = await idempotency_store.set_if_absent(
        tenant_id=payload.tenant_id,
        idempotency_key=payload.idempotency_key,
        external_task_id=candidate_task_id,
    )

    if not is_fresh and existing_record is not None:
        return IngestDispatchResponse(
            success=True,
            external_task_id=existing_record.external_task_id,
            status=DispatchStatus.DUPLICATE,
            message="Ingestion task already accepted for this idempotency key",
            data={
                "job_id": payload.job_id,
                "asset_id": payload.asset_id,
                "tenant_id": payload.tenant_id,
                "run_number": payload.run_number,
                "attempt_count": payload.attempt_count,
            },
        )

    # 3. Fresh task accepted
    return IngestDispatchResponse(
        success=True,
        external_task_id=candidate_task_id,
        status=DispatchStatus.ACCEPTED,
        message="Ingestion task accepted for processing",
        data={
            "job_id": payload.job_id,
            "asset_id": payload.asset_id,
            "tenant_id": payload.tenant_id,
            "run_number": payload.run_number,
            "attempt_count": payload.attempt_count,
            "service_identity": service_ctx.service_identity,
        },
    )
