"""AssetFetcher with a fake S3 client: every outcome is typed, and no
temporary file survives any of them (P3-1 PR 4)."""

import asyncio
import hashlib
import os
import time

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from app.services.ingestion.asset_fetcher import AssetFetcher
from app.services.tasks.ingestion_outcomes import PermanentIngestionError, TransientIngestionError

BUCKET = "build-documents"
TENANT = "tenant-1"
ASSET = "asset-1"
PDF = b"%PDF-1.7\n" + b"lecture notes " * 5000


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeBody:
    def __init__(self, data: bytes, delay: float = 0.0) -> None:
        self._data = data
        self._delay = delay
        self.closed = False

    def iter_chunks(self, size: int):
        for i in range(0, len(self._data), size):
            if self._delay:
                time.sleep(self._delay)
            yield self._data[i : i + size]

    def close(self) -> None:
        self.closed = True


class FakeS3:
    def __init__(self, objects=None, error=None, declared=None, delay=0.0) -> None:
        self.objects = objects or {}
        self.error = error
        self.declared = declared
        self.delay = delay
        self.calls: list[str] = []
        self.bodies: list[FakeBody] = []

    def get_object(self, *, Bucket, Key):  # noqa: N803 (boto3's names)
        self.calls.append(Key)
        if self.error is not None:
            raise self.error
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        data = self.objects[Key]
        body = FakeBody(data, self.delay)
        self.bodies.append(body)
        return {
            "Body": body,
            "ContentLength": self.declared if self.declared is not None else len(data),
        }


def key_for(data: bytes) -> str:
    return f"tenants/{TENANT}/assets/{ASSET}/{sha(data)}"


def uri_for(data: bytes) -> str:
    return f"s3://{BUCKET}/{key_for(data)}"


@pytest.fixture
def temp_dir(tmp_path):
    return str(tmp_path)


def fetcher(client, temp_dir, max_bytes=10 * 1024 * 1024, timeout=5.0) -> AssetFetcher:
    return AssetFetcher(
        bucket=BUCKET,
        client_factory=lambda: client,
        max_bytes=max_bytes,
        timeout_seconds=timeout,
        temp_dir=temp_dir,
    )


def args(data: bytes, **overrides) -> dict:
    return {
        "tenant_id": TENANT,
        "asset_id": ASSET,
        "storage_uri": uri_for(data),
        "content_sha256": sha(data),
        "mime_type": "application/pdf",
    } | overrides


async def test_a_verified_asset_is_available_inside_and_removed_after(temp_dir):
    client = FakeS3({key_for(PDF): PDF})

    async with fetcher(client, temp_dir).fetch(**args(PDF, file_size_bytes=len(PDF))) as asset:
        with open(asset.path, "rb") as f:
            assert f.read() == PDF
        assert (asset.size_bytes, asset.sha256, asset.mime_type) == (
            len(PDF),
            sha(PDF),
            "application/pdf",
        )

    assert os.listdir(temp_dir) == []
    assert client.bodies[0].closed


async def test_a_foreign_reference_never_reaches_storage(temp_dir):
    client = FakeS3({key_for(PDF): PDF})

    with pytest.raises(PermanentIngestionError) as caught:
        async with fetcher(client, temp_dir).fetch(**args(PDF, tenant_id="tenant-2")):
            pass

    assert caught.value.code == "ASSET_REFERENCE_REJECTED"
    assert client.calls == []
    assert os.listdir(temp_dir) == []


@pytest.mark.parametrize(
    ("case", "code", "retryable"),
    [
        ("missing", "ASSET_NOT_FOUND", False),
        ("tampered", "ASSET_INTEGRITY_MISMATCH", False),
        ("wrong size", "ASSET_INTEGRITY_MISMATCH", False),
        ("declared too large", "ASSET_TOO_LARGE", False),
        ("streamed too large", "ASSET_TOO_LARGE", False),
        ("not a pdf", "ASSET_TYPE_MISMATCH", False),
        ("denied", "OBJECT_STORAGE_ACCESS_DENIED", False),
        ("no bucket", "OBJECT_STORAGE_MISCONFIGURED", False),
        ("unreachable", "OBJECT_STORAGE_UNAVAILABLE", True),
        ("throttled", "OBJECT_STORAGE_UNAVAILABLE", True),
        ("server error", "OBJECT_STORAGE_UNAVAILABLE", True),
    ],
)
async def test_every_failure_is_typed_and_leaves_no_file(temp_dir, case, code, retryable):
    data, objects, kwargs, client_kwargs = PDF, {key_for(PDF): PDF}, {}, {}
    max_bytes = 10 * 1024 * 1024
    if case == "missing":
        objects = {}
    elif case == "tampered":
        objects = {key_for(PDF): PDF[:-1] + b"X"}  # same size, other bytes
    elif case == "wrong size":
        kwargs = {"file_size_bytes": len(PDF) + 1}
    elif case == "declared too large":
        client_kwargs = {"declared": max_bytes + 1}
    elif case == "streamed too large":
        max_bytes = 1024
        client_kwargs = {"declared": 10}  # the header lies; the stream is counted
    elif case == "not a pdf":
        data = b"<html>" + b"x" * 100
        objects = {key_for(data): data}
    elif case == "denied":
        client_kwargs = {"error": ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")}
    elif case == "no bucket":
        client_kwargs = {"error": ClientError({"Error": {"Code": "NoSuchBucket"}}, "GetObject")}
    elif case == "unreachable":
        client_kwargs = {"error": EndpointConnectionError(endpoint_url="http://storage")}
    elif case == "throttled":
        client_kwargs = {"error": ClientError({"Error": {"Code": "SlowDown"}}, "GetObject")}
    elif case == "server error":
        client_kwargs = {
            "error": ClientError(
                {"Error": {}, "ResponseMetadata": {"HTTPStatusCode": 500}}, "GetObject"
            )
        }
    client = FakeS3(objects, **client_kwargs)

    with pytest.raises((PermanentIngestionError, TransientIngestionError)) as caught:
        async with fetcher(client, temp_dir, max_bytes=max_bytes).fetch(**args(data, **kwargs)):
            pass

    assert (caught.value.code, caught.value.retryable) == (code, retryable)
    assert os.listdir(temp_dir) == []


async def test_a_dispatched_size_over_the_limit_is_refused_without_a_request(temp_dir):
    client = FakeS3({key_for(PDF): PDF})

    with pytest.raises(PermanentIngestionError) as caught:
        async with fetcher(client, temp_dir, max_bytes=2048).fetch(
            **args(PDF, file_size_bytes=4096)
        ):
            pass

    assert caught.value.code == "ASSET_TOO_LARGE"
    assert client.calls == []


async def test_a_slow_fetch_times_out_as_retryable_and_leaves_no_file(temp_dir):
    client = FakeS3({key_for(PDF): PDF}, delay=0.2)  # 1 MiB chunks; PDF is one chunk

    with pytest.raises(TransientIngestionError) as caught:
        async with fetcher(client, temp_dir, timeout=0.05).fetch(**args(PDF)):
            pass

    assert caught.value.code == "ASSET_FETCH_TIMEOUT"
    assert os.listdir(temp_dir) == []


async def test_the_file_is_removed_when_the_caller_fails_or_is_cancelled(temp_dir):
    client = FakeS3({key_for(PDF): PDF})

    with pytest.raises(RuntimeError):
        async with fetcher(client, temp_dir).fetch(**args(PDF)):
            raise RuntimeError("extraction failed")
    assert os.listdir(temp_dir) == []

    async def cancelled_inside():
        async with fetcher(client, temp_dir).fetch(**args(PDF)):
            await asyncio.sleep(10)

    task = asyncio.create_task(cancelled_inside())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert os.listdir(temp_dir) == []


async def test_missing_credentials_are_a_permanent_configuration_failure(temp_dir):
    from app.core.config import settings

    config = settings.model_copy(
        update={
            "OBJECT_STORAGE_BUCKET": BUCKET,
            "OBJECT_STORAGE_ACCESS_KEY_ID": None,
            "OBJECT_STORAGE_SECRET_ACCESS_KEY": None,
        }
    )
    unconfigured = AssetFetcher.from_settings(config)
    unconfigured._temp_dir = temp_dir

    with pytest.raises(PermanentIngestionError) as caught:
        async with unconfigured.fetch(**args(PDF)):
            pass

    assert caught.value.code == "OBJECT_STORAGE_NOT_CONFIGURED"
    assert os.listdir(temp_dir) == []
