"""Route-level access to a verified backend call (design §A.4, §A.5 step 9).

``SignedServiceRequestMiddleware`` authenticates the request; a route then
declares the one permission it needs with ``signed_call_with(permission)``.
The dependency runs before body validation, so a caller without the
permission gets 403 whatever its body contains.
"""

from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request, status

from app.core.middleware.service_auth import VERIFIED_CALL_STATE
from app.core.service_context import PermissionNotGrantedError, require_permission
from app.core.service_signature_v1 import VerifiedCall
from app.core.tenant import set_request_scope


def signed_call_with(permission: str) -> Callable[[Request], VerifiedCall]:
    if not permission:
        raise ValueError("a signed route must declare one permission")

    def dependency(request: Request) -> VerifiedCall:
        call = getattr(request.state, VERIFIED_CALL_STATE, None)
        if not isinstance(call, VerifiedCall):
            # The route is not behind the middleware: refuse rather than trust.
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid service authentication",
            )
        try:
            require_permission(call.context, permission)
        except PermissionNotGrantedError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="permission not granted"
            ) from exc
        return call

    return dependency


def signed_dispatch_call(permission: str) -> Callable[[Request], VerifiedCall]:
    """Ingestion dispatch (P3-6 §9): a user context holding ``permission``,
    or a system context, the backend worker delivering a job an authorized
    user already created. The task's tenant is the verified context's either
    way. Dispatch and task status are the only routes a system context may
    use besides callbacks; every other route refuses it."""
    check = signed_call_with(permission)

    def dependency(request: Request) -> VerifiedCall:
        call = getattr(request.state, VERIFIED_CALL_STATE, None)
        if isinstance(call, VerifiedCall) and call.context.is_system:
            return call
        return check(request)

    return dependency


def signed_user_call_with(permission: str) -> Callable[[Request], Awaitable[VerifiedCall]]:
    """``signed_call_with(permission)`` for a route acting for a user (design
    B.4): the context must be a user context, and its tenant and user become
    the request scope that memory and retrieval read. Async so that the scope
    is set in the endpoint's own context."""
    check = signed_call_with(permission)

    async def dependency(request: Request) -> VerifiedCall:
        call = check(request)
        if call.context.is_system or not call.context.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="a user context is required"
            )
        set_request_scope(call.context.tenant_id, call.context.user_id)
        return call

    return dependency


def signed_system_call(request: Request) -> VerifiedCall:
    """A verified backend call made on the backend's own authority (a system
    context: tenant only, no user, no permissions), such as reconciliation.
    A user context is refused, whatever permissions it carries."""
    call = getattr(request.state, VERIFIED_CALL_STATE, None)
    if not isinstance(call, VerifiedCall):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid service authentication",
        )
    if not call.context.is_system:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="a system context is required"
        )
    return call
