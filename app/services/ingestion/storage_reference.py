"""Which storage references the AI-service will fetch (P3-1 PR 4).

The backend writes a document at one of two server-generated locations:

    s3://{bucket}/tenants/{tenant}/assets/{asset_id}/{sha256}
    s3://{bucket}/tenants/{tenant}/blobs/{sha256}/{generation}

The first is a document's own object. The second (M3-D PR 7) is a blob that
documents with the same bytes share: it names no document, and its
generation is a lowercase UUID the backend gives each blob row, so a key is
never reused by a later row. The fetch still verifies the bytes against the
dispatched SHA-256 and size, so naming a blob gives access to nothing but
the bytes the signed dispatch already describes.

A dispatched ``storage_uri`` is accepted only if it is exactly one of those
locations for the configured bucket, the tenant of the verified service
context, and the dispatched asset id (for the first) and SHA-256. Anything else, however
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
_GENERATION = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def asset_object_key(tenant_id: str, asset_id: str, sha256: str) -> str:
    return f"tenants/{tenant_id}/assets/{asset_id}/{sha256}"


def _blob_key(uri: str, *, bucket: str, tenant_id: str, content_sha256: str) -> str | None:
    """The key, if ``uri`` is exactly this tenant's blob for this SHA-256."""
    prefix = f"s3://{bucket}/tenants/{tenant_id}/blobs/{content_sha256}/"
    if not uri.startswith(prefix):
        return None
    generation = uri[len(prefix) :]
    if not _GENERATION.fullmatch(generation):
        return None
    return f"tenants/{tenant_id}/blobs/{content_sha256}/{generation}"


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
    if storage_uri == f"s3://{bucket}/{key}":
        return key
    blob = _blob_key(storage_uri, bucket=bucket, tenant_id=tenant_id, content_sha256=content_sha256)
    if blob is not None:
        return blob
    raise PermanentIngestionError(
        REFERENCE_REJECTED,
        "the storage reference is not this tenant's managed object for this asset",
    )
