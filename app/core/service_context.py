"""Signed service context — AI-service side.

Specification: build-backend ``docs/phase1-m3-boundary-design.md`` §A.4.

The context names the tenant (and, for user calls, the user and membership)
a backend request acts for, plus the permissions the backend already checked
for this one operation. AI-service only ever *narrows* behaviour with it:

* scope retrieval and memory to ``tenant_id`` / ``user_id``;
* refuse a route whose single declared permission is not in ``permissions``.

It never derives permissions from ``role`` (kept for audit only), never
expands hierarchies or wildcards, and never treats one permission as implying
another. Resolving roles to permissions is the backend's job alone.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

CONTEXT_SCHEMA_VERSION = 1
USER_ACTOR = "user"
SYSTEM_ACTOR = "system"

_REQUIRED_KEYS = (
    "actor",
    "correlation_id",
    "membership_id",
    "permissions",
    "role",
    "tenant_id",
    "user_id",
    "v",
)
_CANONICAL_UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
_CORRELATION_ID = re.compile(r"[A-Za-z0-9._:\-]{1,128}")


class ServiceContextError(ValueError):
    """The decoded context breaks §A.4. Detail is for internal logs only."""


class PermissionNotGrantedError(Exception):
    """The route's declared permission is not in the signed context (403)."""

    def __init__(self, required_permission: str) -> None:
        super().__init__(f"context lacks permission {required_permission!r}")
        self.required_permission = required_permission


def _uuid(value: object, key: str) -> str:
    if isinstance(value, str) and _CANONICAL_UUID.fullmatch(value):
        return value
    raise ServiceContextError(f"{key}: expected a canonical lowercase UUID")


@dataclass(frozen=True)
class ServiceContext:
    actor: str
    tenant_id: str
    user_id: str | None
    membership_id: str | None
    role: str | None
    permissions: frozenset[str]
    permission_list: tuple[str, ...]
    correlation_id: str

    @property
    def is_system(self) -> bool:
        return self.actor == SYSTEM_ACTOR

    @classmethod
    def system(cls, *, tenant_id: str, correlation_id: str) -> ServiceContext:
        """Context for AI-service-originated calls (callbacks): tenant only,
        no user, no permissions."""
        return parse_service_context(
            {
                "v": CONTEXT_SCHEMA_VERSION,
                "actor": SYSTEM_ACTOR,
                "tenant_id": tenant_id,
                "user_id": None,
                "membership_id": None,
                "role": None,
                "permissions": [],
                "correlation_id": correlation_id,
            }
        )

    def to_json_object(self) -> dict[str, Any]:
        return {
            "v": CONTEXT_SCHEMA_VERSION,
            "actor": self.actor,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "membership_id": self.membership_id,
            "role": self.role,
            "permissions": list(self.permission_list),
            "correlation_id": self.correlation_id,
        }

    def has_permission(self, required_permission: str) -> bool:
        """Exact string membership — nothing else."""
        return required_permission in self.permissions


def require_permission(context: ServiceContext, required_permission: str) -> None:
    """Refuse unless ``required_permission`` is literally in the context.

    Each route declares exactly one permission string. ``role`` is ignored,
    and no permission implies another.
    """
    if not isinstance(required_permission, str) or not required_permission:
        raise ValueError("a route must declare one non-empty permission string")
    if not context.has_permission(required_permission):
        raise PermissionNotGrantedError(required_permission)


def parse_service_context(data: Mapping[str, Any]) -> ServiceContext:
    """Validate a decoded context object against §A.4."""
    present = sorted(data.keys())
    if present != list(_REQUIRED_KEYS):
        missing = [k for k in _REQUIRED_KEYS if k not in data]
        extra = [k for k in present if k not in _REQUIRED_KEYS]
        raise ServiceContextError(f"context keys wrong (missing={missing}, unexpected={extra})")

    version = data["v"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ServiceContextError("v: expected integer 1")

    tenant_id = _uuid(data["tenant_id"], "tenant_id")

    correlation_id = data["correlation_id"]
    if not isinstance(correlation_id, str) or not _CORRELATION_ID.fullmatch(correlation_id):
        raise ServiceContextError("correlation_id: expected 1-128 chars of [A-Za-z0-9._:-]")

    raw_permissions = data["permissions"]
    if not isinstance(raw_permissions, list):
        raise ServiceContextError("permissions: expected an array")
    for item in raw_permissions:
        if not isinstance(item, str) or item == "":
            raise ServiceContextError("permissions: expected non-empty strings")
    for earlier, later in zip(raw_permissions, raw_permissions[1:]):
        if not earlier < later:
            raise ServiceContextError("permissions: expected strictly ascending order")

    actor = data["actor"]
    user_id: str | None
    membership_id: str | None
    role: str | None
    if actor == USER_ACTOR:
        user_id = _uuid(data["user_id"], "user_id")
        membership_id = _uuid(data["membership_id"], "membership_id")
        role = data["role"]
        if not isinstance(role, str) or role == "":
            raise ServiceContextError("role: expected a non-empty string")
    elif actor == SYSTEM_ACTOR:
        for key in ("user_id", "membership_id", "role"):
            if data[key] is not None:
                raise ServiceContextError(f"{key}: must be null for the system actor")
        if raw_permissions:
            raise ServiceContextError("permissions: must be empty for the system actor")
        user_id = membership_id = role = None
    else:
        raise ServiceContextError("actor: expected 'user' or 'system'")

    return ServiceContext(
        actor=actor,
        tenant_id=tenant_id,
        user_id=user_id,
        membership_id=membership_id,
        role=role,
        permissions=frozenset(raw_permissions),
        permission_list=tuple(raw_permissions),
        correlation_id=correlation_id,
    )
