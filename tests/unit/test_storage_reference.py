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
