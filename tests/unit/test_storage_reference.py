"""resolve_storage_reference: only the exact managed object of the verified
tenant and the dispatched asset is fetched (P3-1 PR 4)."""

import hashlib

import pytest

from app.services.ingestion.storage_reference import REFERENCE_REJECTED, resolve_storage_reference
from app.services.tasks.ingestion_outcomes import PermanentIngestionError

BUCKET = "build-documents"
TENANT = "6f1c2a3b-0d4e-4f5a-9b6c-7d8e9f0a1b2c"
ASSET = "0b9a8c7d-6e5f-5a4b-8c3d-2e1f0a9b8c7d"
SHA = hashlib.sha256(b"notes").hexdigest()
GOOD = f"s3://{BUCKET}/tenants/{TENANT}/assets/{ASSET}/{SHA}"


def resolve(uri: str, **overrides) -> str:
    args = {
        "bucket": BUCKET,
        "tenant_id": TENANT,
        "asset_id": ASSET,
        "content_sha256": SHA,
    } | overrides
    return resolve_storage_reference(uri, **args)


def test_the_managed_object_is_resolved_to_its_key() -> None:
    assert resolve(GOOD) == f"tenants/{TENANT}/assets/{ASSET}/{SHA}"


@pytest.mark.parametrize(
    "uri",
    [
        GOOD.replace(BUCKET, "other-bucket"),  # foreign bucket
        GOOD.replace(TENANT, "11111111-2222-4333-8444-555555555555"),  # another tenant's prefix
        GOOD.replace(ASSET, "11111111-2222-5333-8444-555555555555"),  # another asset's object
        GOOD.replace(SHA, "0" * 64),  # other bytes
        GOOD + "/extra",
        GOOD + "/",
        GOOD.replace("/tenants/", "//tenants/"),
        GOOD.replace(f"/assets/{ASSET}", f"/assets/x/../{ASSET}"),
        GOOD.replace(SHA, SHA.upper()),
        GOOD.replace("s3://", "S3://"),
        GOOD.replace("s3://", "https://"),
        GOOD.replace(ASSET, ASSET.replace("-", "%2D")),
        f"tenants/{TENANT}/assets/{ASSET}/{SHA}",  # a bare key
        f"http://169.254.169.254/latest/meta-data/{SHA}",  # an arbitrary URL
        f"file:///etc/{SHA}",
        GOOD + " ",
    ],
)
def test_any_other_reference_is_rejected_before_storage_is_contacted(uri: str) -> None:
    with pytest.raises(PermanentIngestionError) as caught:
        resolve(uri)

    assert caught.value.code == REFERENCE_REJECTED
    assert uri not in str(caught.value)  # the reference is never echoed


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenant_id": "../other"},
        {"tenant_id": ""},
        {"asset_id": "a/b"},
        {"asset_id": "a.b"},
        {"content_sha256": "z" * 64},
        {"bucket": ""},
    ],
)
def test_malformed_identity_or_missing_configuration_is_rejected(overrides) -> None:
    with pytest.raises(PermanentIngestionError) as caught:
        resolve(GOOD, **overrides)

    assert caught.value.code == REFERENCE_REJECTED


# --- M3-D PR 7: the blob form ---------------------------------------------------------

GENERATION = "3f2c1b0a-9d8e-4c7b-a6f5-e4d3c2b1a098"
BLOB = f"s3://{BUCKET}/tenants/{TENANT}/blobs/{SHA}/{GENERATION}"


def test_the_tenants_blob_for_the_dispatched_hash_is_resolved_to_its_key() -> None:
    assert resolve(BLOB) == f"tenants/{TENANT}/blobs/{SHA}/{GENERATION}"


def test_a_blob_names_no_document_so_any_asset_of_the_tenant_may_share_it() -> None:
    other_asset = "11111111-2222-5333-8444-555555555555"
    assert resolve(BLOB, asset_id=other_asset) == f"tenants/{TENANT}/blobs/{SHA}/{GENERATION}"


@pytest.mark.parametrize(
    "uri",
    [
        BLOB.replace(BUCKET, "other-bucket"),
        BLOB.replace(TENANT, "11111111-2222-4333-8444-555555555555"),  # another tenant's blob
        BLOB.replace(SHA, "0" * 64),  # another hash's blob
        BLOB.replace(SHA, SHA.upper()),
        BLOB.replace(GENERATION, GENERATION.upper()),
        BLOB.replace(GENERATION, GENERATION.replace("-", "")),
        BLOB.replace(GENERATION, GENERATION[:-1]),
        BLOB.replace(GENERATION, GENERATION + "0"),
        BLOB.replace(GENERATION, "../" + GENERATION),
        BLOB.replace(GENERATION, "x/" + GENERATION),
        BLOB.replace(GENERATION, GENERATION.replace("-", "%2D")),
        BLOB.replace(f"/{GENERATION}", ""),  # no generation
        BLOB.replace(f"/{GENERATION}", "/"),
        BLOB + "/",
        BLOB + "/extra",
        BLOB + " ",
        BLOB + "\n",
        BLOB.replace("/blobs/", "/Blobs/"),
        BLOB.replace("/blobs/", "//blobs/"),
        BLOB.replace("/blobs/", "/assets/"),  # the asset form with a generation for an id
        BLOB.replace("s3://", "https://"),
        f"tenants/{TENANT}/blobs/{SHA}/{GENERATION}",  # a bare key
    ],
)
def test_any_other_blob_reference_is_rejected(uri: str) -> None:
    with pytest.raises(PermanentIngestionError) as caught:
        resolve(uri)

    assert caught.value.code == REFERENCE_REJECTED
    assert uri.strip() not in str(caught.value)


@pytest.mark.parametrize(
    "overrides",
    [{"tenant_id": "../other"}, {"content_sha256": "z" * 64}, {"bucket": ""}],
)
def test_a_blob_with_malformed_identity_or_no_bucket_is_rejected(overrides) -> None:
    with pytest.raises(PermanentIngestionError) as caught:
        resolve(BLOB, **overrides)

    assert caught.value.code == REFERENCE_REJECTED
