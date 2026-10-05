"""Route-level access to a verified backend call (design §A.4, §A.5 step 9).

``SignedServiceRequestMiddleware`` authenticates the request; a route then
declares the one permission it needs with ``signed_call_with(permission)``.
The dependency runs before body validation, so a caller without the
permission gets 403 whatever its body contains.
"""

from collections.abc import Callable

from fastapi import HTTPException, Request, status

from app.core.middleware.service_auth import VERIFIED_CALL_STATE
from app.core.service_context import PermissionNotGrantedError, require_permission
from app.core.service_signature_v1 import VerifiedCall


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
