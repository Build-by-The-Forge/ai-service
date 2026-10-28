import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.idempotency import (
    IdempotencyStore,
    IdempotencyUnavailableError,
    get_idempotency_store,
)
from app.core.service_call import signed_call_with, signed_system_call
from app.core.service_signature_v1 import VerifiedCall
from app.schemas.ingest import (
    DispatchStatus,
    IngestDispatchRequest,
    IngestDispatchResponse,
    IngestTaskStatusResponse,
)
from app.services.tasks.dispatcher import TaskDispatcher, get_task_dispatcher
from app.services.tasks.task_records import (
    TaskRecord,
    TaskRecordStore,
    TaskRecordUnavailableError,
    get_task_record_store,
    reported_state,
)

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
    task_records: TaskRecordStore = Depends(get_task_record_store),
) -> IngestDispatchResponse:
    tenant_id = call.context.tenant_id
    receipt = {
        "job_id": payload.job_id,
        "asset_id": payload.asset_id,
        "run_number": payload.run_number,
        "attempt_count": payload.attempt_count,
    }

    candidate_task_id = f"task_{uuid.uuid4().hex}"
    try:
        is_fresh, existing_record = await idempotency_store.set_if_absent(
            tenant_id=tenant_id,
            idempotency_key=payload.idempotency_key,
            external_task_id=candidate_task_id,
        )
    except IdempotencyUnavailableError as exc:
        # Fail closed: never dispatch without a shared idempotency decision.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service temporarily unavailable",
        ) from exc

    if not is_fresh and existing_record is not None:
        return IngestDispatchResponse(
            success=True,
            external_task_id=existing_record.external_task_id,
            status=DispatchStatus.DUPLICATE,
            message="Ingestion task already accepted for this idempotency key",
            data=receipt,
        )

    # The record ties the task to the verified tenant and backend attempt;
    # status reconciliation (M3-09) answers only from it.
    try:
        await task_records.create(
            TaskRecord(
                external_task_id=candidate_task_id,
                tenant_id=tenant_id,
                job_id=payload.job_id,
                run_number=payload.run_number,
                attempt_count=payload.attempt_count,
            )
        )
    except TaskRecordUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service temporarily unavailable",
        ) from exc

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


@router.get(
    "/tasks/{external_task_id}",
    response_model=IngestTaskStatusResponse,
    summary="Ingestion Task Status",
    description=(
        "Signed backend-only status of one task, for reconciliation. Requires a system "
        "context; answers only for a task of the context's tenant (404 otherwise)."
    ),
)
async def get_ingest_task(
    external_task_id: str,
    call: VerifiedCall = Depends(signed_system_call),
    task_records: TaskRecordStore = Depends(get_task_record_store),
) -> IngestTaskStatusResponse:
    try:
        record = await task_records.get(external_task_id)
    except TaskRecordUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service temporarily unavailable",
        ) from exc
    # Unknown and other tenants' tasks look the same.
    if record is None or record.tenant_id != call.context.tenant_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task not found")
    return IngestTaskStatusResponse.model_validate(
        {
            "external_task_id": record.external_task_id,
            "job_id": record.job_id,
            "run_number": record.run_number,
            "attempt_count": record.attempt_count,
            "state": reported_state(record),
            "events": record.events,
            "lease_until": record.lease_until,
            "heartbeat_at": record.heartbeat_at,
        }
    )
