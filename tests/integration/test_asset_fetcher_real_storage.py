"""AssetFetcher against a real S3-compatible service (SeaweedFS in CI, from
scripts/seaweedfs_test_storage.sh), with the AI's read-only identity
(P3-1 PR 4).

The writer seeds objects as build-backend does; the fetcher reads with the
reader identity, which may only read bucket A. With
OBJECT_STORAGE_TESTS_REQUIRED=1, as in CI, missing configuration is a
collection error rather than a skip.
"""

import hashlib
import os
import uuid

import boto3
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError

from app.core.config import settings
from app.services.ingestion.asset_fetcher import AssetFetcher
from app.services.tasks.ingestion_outcomes import PermanentIngestionError

_ENV = "OBJECT_STORAGE_TEST_"
_NEEDED = ("ENDPOINT", "BUCKET_A", "BUCKET_B", "WRITER_A", "READER_A", "WRITER_B")
CONFIG = {name: os.getenv(_ENV + name) for name in (*_NEEDED, "REGION")}
CONFIGURED = all(CONFIG[name] for name in _NEEDED)

if os.getenv("OBJECT_STORAGE_TESTS_REQUIRED") == "1" and not CONFIGURED:
    missing = ", ".join(_ENV + name for name in _NEEDED if not CONFIG[name])
    raise RuntimeError(f"real object-storage tests are required, but {missing} are not set")

pytestmark = pytest.mark.skipif(not CONFIGURED, reason="OBJECT_STORAGE_TEST_* not set")

TENANT = str(uuid.uuid4())
PDF = b"%PDF-1.7\n" + b"real storage lecture notes " * 4000


def _client(identity: str):
    access, secret = CONFIG[identity].split(":", 1)
    return boto3.session.Session().client(
        "s3",
        endpoint_url=CONFIG["ENDPOINT"],
        region_name=CONFIG["REGION"] or "us-east-1",
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        config=Config(
            s3={"addressing_style": "path"},
            retries={"max_attempts": 1},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


@pytest.fixture
def writer():
    return _client("WRITER_A")


@pytest.fixture
def fetcher(tmp_path):
    reader = _client("READER_A")
    return AssetFetcher(
        bucket=CONFIG["BUCKET_A"],
        client_factory=lambda: reader,
        max_bytes=settings.ASSET_FETCH_MAX_BYTES,
        timeout_seconds=30,
        temp_dir=str(tmp_path),
    )


@pytest.fixture
def stored(writer):
    """One object at its managed key, removed afterwards."""
    asset_id = str(uuid.uuid4())
    sha = hashlib.sha256(PDF).hexdigest()
    key = f"tenants/{TENANT}/assets/{asset_id}/{sha}"
    writer.put_object(Bucket=CONFIG["BUCKET_A"], Key=key, Body=PDF, ContentType="application/pdf")
    yield {"asset_id": asset_id, "sha": sha, "key": key}
    writer.delete_object(Bucket=CONFIG["BUCKET_A"], Key=key)


def _args(stored) -> dict:
    return {
        "tenant_id": TENANT,
        "asset_id": stored["asset_id"],
        "storage_uri": f"s3://{CONFIG['BUCKET_A']}/{stored['key']}",
        "content_sha256": stored["sha"],
        "mime_type": "application/pdf",
        "file_size_bytes": len(PDF),
    }


async def test_the_read_only_identity_fetches_and_verifies_a_stored_document(
    fetcher, stored, tmp_path
):
    async with fetcher.fetch(**_args(stored)) as asset:
        with open(asset.path, "rb") as f:
            assert f.read() == PDF

    assert list(tmp_path.iterdir()) == []


async def test_a_tampered_object_of_the_same_size_fails_permanently(fetcher, stored, writer):
    writer.put_object(Bucket=CONFIG["BUCKET_A"], Key=stored["key"], Body=PDF[:-1] + b"X")

    with pytest.raises(PermanentIngestionError) as caught:
        async with fetcher.fetch(**_args(stored)):
            pass

    assert caught.value.code == "ASSET_INTEGRITY_MISMATCH"


async def test_a_missing_object_fails_permanently(fetcher, stored, writer):
    writer.delete_object(Bucket=CONFIG["BUCKET_A"], Key=stored["key"])

    with pytest.raises(PermanentIngestionError) as caught:
        async with fetcher.fetch(**_args(stored)):
            pass

    assert caught.value.code == "ASSET_NOT_FOUND"


def test_the_ai_identity_cannot_write_or_delete():
    reader = _client("READER_A")
    key = f"tenants/{TENANT}/assets/{uuid.uuid4()}/{'0' * 64}"

    for call in (
        lambda: reader.put_object(Bucket=CONFIG["BUCKET_A"], Key=key, Body=b"%PDF-1.7 x"),
        lambda: reader.delete_object(Bucket=CONFIG["BUCKET_A"], Key=key),
    ):
        with pytest.raises(ClientError) as caught:
            call()
        assert caught.value.response["Error"]["Code"] == "AccessDenied"


async def test_another_bucket_is_refused_before_storage(fetcher, stored):
    args = _args(stored) | {"storage_uri": f"s3://{CONFIG['BUCKET_B']}/{stored['key']}"}

    with pytest.raises(PermanentIngestionError) as caught:
        async with fetcher.fetch(**args):
            pass

    assert caught.value.code == "ASSET_REFERENCE_REJECTED"
