import uuid

from fastapi import APIRouter, Depends, status

from app.core.idempotency import IdempotencyStore, get_idempotency_store
from app.core.service_call import signed_call_with
from app.core.service_signature_v1 import VerifiedCall
from app.schemas.ingest import (
    DispatchStatus,
    IngestDispatchRequest,
    IngestDispatchResponse,
)
from app.services.tasks.dispatcher import TaskDispatcher, get_task_dispatcher

router = APIRouter()

# The backend checks this permission before dispatching (design B.2) and
# records it in the signed context; AI-service requires exactly this string.
INGEST_DISPATCH_PERMISSION = "content.job.manage"


@router.post(
    "/dispatch",
    response_model=IngestDispatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Dispatch Content Ingestion Task",
    description=(
        "Signed backend-only endpoint (BUILD Service Signature v1). The task's tenant is "
        "the verified context's tenant; the body cannot name one. Idempotent per tenant "
        "and idempotency key."
    ),
)
async def dispatch_ingest_task(
    payload: IngestDispatchRequest,
    call: VerifiedCall = Depends(signed_call_with(INGEST_DISPATCH_PERMISSION)),
    idempotency_store: IdempotencyStore = Depends(get_idempotency_store),
    dispatcher: TaskDispatcher = Depends(get_task_dispatcher),
) -> IngestDispatchResponse:
    tenant_id = call.context.tenant_id
    receipt = {
        "job_id": payload.job_id,
        "asset_id": payload.asset_id,
        "run_number": payload.run_number,
        "attempt_count": payload.attempt_count,
    }

    candidate_task_id = f"task_{uuid.uuid4().hex}"
    is_fresh, existing_record = await idempotency_store.set_if_absent(
        tenant_id=tenant_id,
        idempotency_key=payload.idempotency_key,
        external_task_id=candidate_task_id,
    )

    if not is_fresh and existing_record is not None:
        return IngestDispatchResponse(
            success=True,
            external_task_id=existing_record.external_task_id,
            status=DispatchStatus.DUPLICATE,
            message="Ingestion task already accepted for this idempotency key",
            data=receipt,
        )

    await dispatcher.dispatch_ingest(
        job_id=payload.job_id,
        asset_id=payload.asset_id,
        tenant_id=tenant_id,
        correlation_id=payload.correlation_id,
        payload=payload.model_dump(),
        run_number=payload.run_number,
        attempt_count=payload.attempt_count,
        external_task_id=candidate_task_id,
    )

    return IngestDispatchResponse(
        success=True,
        external_task_id=candidate_task_id,
        status=DispatchStatus.ACCEPTED,
        message="Ingestion task accepted for processing",
        data=receipt,
    )
