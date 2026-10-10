"""Request tenant and user scope.

The paths M3 touches (ingestion, intake, retrieval, memory, chat, explain and
question generation) take their tenant and user only from the verified BUILD
service context: the signed route binds them with ``set_request_scope`` and
readers fail closed when nothing was bound. There is no default tenant.

The integer ``school_id`` below survives only inside the legacy user routes
that M4 deletes.
"""

from contextvars import ContextVar

current_tenant_id: ContextVar[str | None] = ContextVar("current_tenant_id", default=None)
current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)


class RequestScopeMissingError(RuntimeError):
    """A tenant- or user-scoped operation ran without a verified scope."""


def set_request_scope(tenant_id: str, user_id: str | None = None) -> None:
    if not tenant_id:
        raise RequestScopeMissingError("a request scope needs a tenant")
    current_tenant_id.set(tenant_id)
    current_user_id.set(user_id or None)


def clear_request_scope() -> None:
    current_tenant_id.set(None)
    current_user_id.set(None)


def get_tenant_id() -> str:
    tenant_id = current_tenant_id.get()
    if not tenant_id:
        raise RequestScopeMissingError("no verified tenant in the request scope")
    return tenant_id


def get_user_id() -> str:
    user_id = current_user_id.get()
    if not user_id:
        raise RequestScopeMissingError("no verified user in the request scope")
    return user_id


# --- Legacy (M4 deletes) ------------------------------------------------------

current_school_id: ContextVar[int] = ContextVar("current_school_id")


def set_school_id(school_id: int):
    current_school_id.set(school_id)


def get_school_id() -> int:
    return current_school_id.get()
