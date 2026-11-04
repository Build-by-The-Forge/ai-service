"""Signed scope assignments (M3-D), mounted at /v1/ingest.

Only the backend sets a document's scope, on its own authority: the route
needs a system context, and a user context is refused whatever permissions
it carries. No student or staff request carries a scope.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.service_call import signed_system_call
from app.core.service_signature_v1 import VerifiedCall
from app.schemas.ingest import ScopeAssignmentRequest, ScopeAssignmentResponse
from app.services.rag.scopes import (
    AssetHeldByAnotherTenantError,
    InMemoryScopes,
    PgScopeStore,
    ScopeStoreUnavailableError,
    get_scope_store,
)

logger = logging.getLogger(__name__)
router = APIRouter()


async def scope_store() -> PgScopeStore | InMemoryScopes:
    return get_scope_store()


@router.post(
    "/scope-assignments",
    response_model=ScopeAssignmentResponse,
    summary="Assign a document's knowledge scope",
    description=(
        "Signed backend-only. Applies the assignment if its version is higher than the "
        "stored one, and returns the scope now stored either way. 409 if another tenant "
        "holds the asset."
    ),
)
async def assign_scope(
    body: ScopeAssignmentRequest,
    call: VerifiedCall = Depends(signed_system_call),
    store: PgScopeStore | InMemoryScopes = Depends(scope_store),
) -> ScopeAssignmentResponse:
    try:
        state = await store.assign(
            tenant_id=call.context.tenant_id,
            asset_id=body.asset_id,
            scope=body.scope,
            version=body.version,
        )
    except AssetHeldByAnotherTenantError as exc:
        logger.warning(
            "scope assignment refused: asset held by another tenant tenant=%s asset=%s",
            call.context.tenant_id,
            body.asset_id,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="asset belongs to another tenant"
        ) from exc
    except ScopeStoreUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="service temporarily unavailable",
        ) from exc
    return ScopeAssignmentResponse(
        asset_id=state.asset_id, scope=state.scope, version=state.version
    )
