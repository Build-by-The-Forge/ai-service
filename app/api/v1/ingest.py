import hashlib
import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.idempotency import (
    IdempotencyStore,
    IdempotencyUnavailableError,
    get_idempotency_store,
)
from app.core.rag.errors import VectorStoreError
from app.core.service_call import signed_dispatch_call, signed_system_call
from app.core.service_signature_v1 import VerifiedCall
from app.schemas.ingest import (
    DispatchStatus,
    IngestDispatchRequest,
    IngestDispatchResponse,
    IngestTaskStatusResponse,
)
from app.services.rag.index_fence import IndexFencePort, get_index_fence
from app.services.tasks.dispatcher import TaskDispatcher, get_task_dispatcher
from app.services.tasks.task_records import (
    TaskRecord,
    TaskRecordStore,
    TaskRecordUnavailableError,
    get_task_record_store,
    reported_state,
)
from app.services.tasks.transport import TaskQueueUnavailableError

logger = logging.getLogger(__name__)
router = APIRouter()

# What identifies the work a dispatch asks for. The correlation id is
# tracing, not work, and the key is what is being compared.
_NOT_FINGERPRINTED = frozenset({"correlation_id", "idempotency_key"})


def request_fingerprint(payload: IngestDispatchRequest) -> str:
    work = {k: v for k, v in payload.model_dump().items() if k not in _NOT_FINGERPRINTED}
    return hashlib.sha256(
        json.dumps(work, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _unavailable(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="service temporarily unavailable",
    )


# The backend checks this permission before dispatching (design B.2) and
# records it in the signed context; AI-service requires exactly this string.
INGEST_DISPATCH_PERMISSION = "content.job.manage"


@router.post(
    "/dispatch",
    response_model=IngestDispatchResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Dispatch Content Ingestion Task",
    description=(
        "Signed backend-only endpoint (BUILD Service Signature v1): a user context with "
        "content.job.manage, or the backend worker's system context. The task's tenant is "
        "the verified context's tenant; the body cannot name one. Idempotent per tenant "
        "and idempotency key. Outcomes: 202 ACCEPTED (queued now) or DUPLICATE (the "
        "original task, queued); 409 the key was used for a different request (no task); "
        "400/401/403/422 refused before any task exists; 503 not decided, retry the same "
        "request (a task exists only if a later retry answers DUPLICATE)."
    ),
)
async def dispatch_ingest_task(
    payload: IngestDispatchRequest,
    call: VerifiedCall = Depends(signed_dispatch_call(INGEST_DISPATCH_PERMISSION)),
    idempotency_store: IdempotencyStore = Depends(get_idempotency_store),
    dispatcher: TaskDispatcher = Depends(get_task_dispatcher),
    task_records: TaskRecordStore = Depends(get_task_record_store),
    index_fence: IndexFencePort = Depends(get_index_fence),
) -> IngestDispatchResponse:
    tenant_id = call.context.tenant_id
    receipt = {
        "job_id": payload.job_id,
        "asset_id": payload.asset_id,
        "run_number": payload.run_number,
        "attempt_count": payload.attempt_count,
    }

    candidate_task_id = f"task_{uuid.uuid4().hex}"
    fingerprint = request_fingerprint(payload)
    try:
        is_fresh, existing_record = await idempotency_store.set_if_absent(
            tenant_id=tenant_id,
            idempotency_key=payload.idempotency_key,
            external_task_id=candidate_task_id,
            fingerprint=fingerprint,
        )
    except IdempotencyUnavailableError as exc:
        # Fail closed: never dispatch without a shared idempotency decision.
        raise _unavailable(exc) from exc

    if not is_fresh and existing_record is not None:
        if existing_record.fingerprint is not None and existing_record.fingerprint != fingerprint:
            # The key names other work: never answer with that work's task.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="IDEMPOTENCY_CONFLICT: the key was used for a different request",
            )
        if not existing_record.accepted:
            # The first request has claimed the key but not yet queued its
            # task: not decided, so the caller retries the same request.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="DISPATCH_IN_PROGRESS: retry the same request",
            )
        return IngestDispatchResponse(
            success=True,
            external_task_id=existing_record.external_task_id,
            status=DispatchStatus.DUPLICATE,
            message="Ingestion task already accepted for this idempotency key",
            data=receipt,
        )

    # Only a request that owns a fresh idempotency claim may advance the
    # asset's fence. In particular, a conflicting reuse of a key must not
    # fence the legitimate request's index writes.
    try:
        await index_fence.raise_fence(
            tenant_id=tenant_id,
            asset_id=payload.asset_id,
            run_number=payload.run_number,
            attempt=payload.attempt_count,
        )
    except VectorStoreError as exc:
        try:
            await idempotency_store.release(tenant_id, payload.idempotency_key, candidate_task_id)
        except IdempotencyUnavailableError:
            logger.warning(
                "Dispatch claim not released after fence failure: task=%s", candidate_task_id
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service temporarily unavailable",
        ) from exc

    # The record ties the task to the verified tenant and backend attempt;
    # status reconciliation (M3-09) answers only from it. If the record or
    # the queue write fails, no task exists: the claim is released, so the
    # same request can claim the key again rather than be told DUPLICATE.
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
    except (TaskRecordUnavailableError, TaskQueueUnavailableError) as exc:
        try:
            await idempotency_store.release(tenant_id, payload.idempotency_key, candidate_task_id)
        except IdempotencyUnavailableError:
            # The claim stays PROCESSING: retries are answered 503 until it
            # expires, never DUPLICATE for a task that was not queued.
            logger.warning("Dispatch claim not released: task=%s", candidate_task_id)
        raise _unavailable(exc) from exc

    try:
        await idempotency_store.mark_accepted(tenant_id, payload.idempotency_key, candidate_task_id)
    except IdempotencyUnavailableError:
        # The task is queued; until the claim is marked, a retry is answered
        # 503 (not decided), never a second task.
        logger.warning("Dispatch claim not marked accepted: task=%s", candidate_task_id)

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
