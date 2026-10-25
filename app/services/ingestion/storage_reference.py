"""Which storage references the AI-service will fetch (P3-1 PR 4).

The backend writes every document at one server-generated location:

    s3://{bucket}/tenants/{tenant}/assets/{asset_id}/{sha256}

A dispatched ``storage_uri`` is accepted only if it is exactly that
location for the configured bucket, the tenant of the verified service
context, and the dispatched asset id and SHA-256. Anything else, however
well formed, is refused before any request reaches storage: another bucket
or tenant, another asset's or hash's object, extra segments, encodings,
case changes, or any other scheme or host (no SSRF path, plan D2 condition
9). The same rule as build-backend's ``managed_storage_key``.
"""

from __future__ import annotations

import re

from app.services.tasks.ingestion_outcomes import PermanentIngestionError

REFERENCE_REJECTED = "ASSET_REFERENCE_REJECTED"

# Ids the backend generates (UUIDs) and tenant ids: no separators, dots,
# percent signs or whitespace can reach a key.
_ID = re.compile(r"[A-Za-z0-9-]{1,64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def asset_object_key(tenant_id: str, asset_id: str, sha256: str) -> str:
    return f"tenants/{tenant_id}/assets/{asset_id}/{sha256}"


def resolve_storage_reference(
    storage_uri: str, *, bucket: str, tenant_id: str, asset_id: str, content_sha256: str
) -> str:
    """The object key to fetch, or PermanentIngestionError
    (ASSET_REFERENCE_REJECTED). The message never echoes the reference."""
    if not bucket:
        raise PermanentIngestionError(REFERENCE_REJECTED, "no object storage bucket is configured")
    if not (
        _ID.fullmatch(tenant_id or "")
        and _ID.fullmatch(asset_id or "")
        and _SHA256.fullmatch(content_sha256 or "")
    ):
        raise PermanentIngestionError(
            REFERENCE_REJECTED, "the tenant, asset id or SHA-256 is not well formed"
        )
    key = asset_object_key(tenant_id, asset_id, content_sha256)
    if storage_uri != f"s3://{bucket}/{key}":
        raise PermanentIngestionError(
            REFERENCE_REJECTED,
            "the storage reference is not this tenant's managed object for this asset",
        )
    return key
