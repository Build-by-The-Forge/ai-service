"""Fetch a dispatched document from object storage and verify it
(P3-1 PR 4, plan section P3-1 "AI fetcher").

``AssetFetcher.fetch`` is an async context manager. Inside it the caller has
a local file holding exactly the bytes that were uploaded:

1. The dispatched ``storage_uri`` must be the verified tenant's managed
   object for the dispatched asset and SHA-256 (``storage_reference``);
   nothing else reaches storage.
2. One GET, streamed in 1 MiB chunks to a temporary file, hashed as it
   arrives, under ``ASSET_FETCH_MAX_BYTES`` (a larger declared length is
   refused before the body is read) and one deadline for the whole fetch.
3. The SHA-256 must equal the dispatched one, the size the dispatched size
   when given, and the file's signature the dispatched MIME type.
4. The temporary file is removed on every exit: success, failure, timeout
   and cancellation.

Failures are typed for the job lifecycle (``ingestion_outcomes``):

- permanent: ASSET_REFERENCE_REJECTED, ASSET_NOT_FOUND,
  ASSET_INTEGRITY_MISMATCH, ASSET_TOO_LARGE, ASSET_TYPE_MISMATCH,
  OBJECT_STORAGE_ACCESS_DENIED, OBJECT_STORAGE_MISCONFIGURED,
  OBJECT_STORAGE_NOT_CONFIGURED;
- transient: OBJECT_STORAGE_UNAVAILABLE, ASSET_FETCH_TIMEOUT.

Logging carries the operation and error code only: never the key, bytes,
credentials or signed URLs.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import os
import tempfile
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, BinaryIO

import boto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    NoCredentialsError,
    ReadTimeoutError,
    ResponseStreamingError,
)
from botocore.exceptions import ConnectionError as BotoConnectionError

from app.core.config import Settings, settings
from app.services.ingestion.storage_reference import resolve_storage_reference
from app.services.tasks.ingestion_outcomes import (
    IngestionFailure,
    PermanentIngestionError,
    TransientIngestionError,
)

logger = logging.getLogger(__name__)
# The SDK must never log request parameters or bodies.
for _name in ("boto3", "botocore", "s3transfer", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

CHUNK_BYTES = 1024 * 1024

# Formats the backend accepts, by signature (its upload allow-list).
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)
_HEAD_BYTES = 8

_NOT_FOUND = frozenset({"NoSuchKey", "NotFound", "404"})
_DENIED = frozenset(
    {"AccessDenied", "403", "InvalidAccessKeyId", "SignatureDoesNotMatch", "AllAccessDisabled"}
)
_MISCONFIGURED = frozenset(
    {"NoSuchBucket", "InvalidBucketName", "PermanentRedirect", "AuthorizationHeaderMalformed"}
)
_TRANSIENT = frozenset(
    {"SlowDown", "ServiceUnavailable", "InternalError", "RequestTimeout", "Throttling", "503"}
)


@dataclass(frozen=True, slots=True)
class FetchedAsset:
    """A verified local copy, valid only inside ``AssetFetcher.fetch``."""

    path: str
    size_bytes: int
    sha256: str
    mime_type: str


class _Stopped(Exception):
    """The fetch was abandoned (timeout or cancellation) while streaming."""


def classify_client_error(error: ClientError) -> IngestionFailure:
    """A provider error as a typed ingestion failure. Unknown errors are
    permanent: never retried blindly, never reported as "not found"."""
    details = error.response.get("Error", {}) or {}
    code = str(details.get("Code") or "")
    status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if code in _NOT_FOUND or status == 404:
        return PermanentIngestionError("ASSET_NOT_FOUND", "the asset's object does not exist")
    if code in _DENIED or status == 403:
        return PermanentIngestionError(
            "OBJECT_STORAGE_ACCESS_DENIED", "object storage refused the AI-service's identity"
        )
    if code in _MISCONFIGURED:
        return PermanentIngestionError(
            "OBJECT_STORAGE_MISCONFIGURED", "the object storage bucket is misconfigured"
        )
    if code in _TRANSIENT or (isinstance(status, int) and (status >= 500 or status == 429)):
        return TransientIngestionError(
            "OBJECT_STORAGE_UNAVAILABLE", "object storage is unavailable"
        )
    return PermanentIngestionError(
        "OBJECT_STORAGE_FAILURE", f"object storage error {code or status}"
    )


def detect_mime_type(head: bytes) -> str | None:
    for signature, mime_type in _SIGNATURES:
        if head.startswith(signature):
            return mime_type
    return None


def _s3_client(config: Settings) -> Any:
    secret = config.OBJECT_STORAGE_SECRET_ACCESS_KEY
    if not config.OBJECT_STORAGE_ACCESS_KEY_ID or secret is None:
        raise PermanentIngestionError(
            "OBJECT_STORAGE_NOT_CONFIGURED", "object storage credentials are not configured"
        )
    return boto3.session.Session().client(
        "s3",
        endpoint_url=config.OBJECT_STORAGE_ENDPOINT_URL or None,
        region_name=config.OBJECT_STORAGE_REGION,
        aws_access_key_id=config.OBJECT_STORAGE_ACCESS_KEY_ID,
        aws_secret_access_key=secret.get_secret_value(),
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": config.OBJECT_STORAGE_ADDRESSING_STYLE},
            connect_timeout=config.OBJECT_STORAGE_CONNECT_TIMEOUT_SECONDS,
            read_timeout=config.OBJECT_STORAGE_READ_TIMEOUT_SECONDS,
            retries={"max_attempts": config.OBJECT_STORAGE_MAX_ATTEMPTS, "mode": "standard"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


class AssetFetcher:
    def __init__(
        self,
        *,
        bucket: str,
        client_factory: Callable[[], Any],
        max_bytes: int,
        timeout_seconds: float,
        temp_dir: str | None = None,
    ) -> None:
        self._bucket = bucket
        self._client_factory = client_factory
        self._max_bytes = max_bytes
        self._timeout_seconds = timeout_seconds
        self._temp_dir = temp_dir

    @classmethod
    def from_settings(cls, config: Settings | None = None) -> AssetFetcher:
        config = config or settings
        # One client per fetcher, built when first needed: boto3 clients are
        # thread-safe and expensive to create.
        return cls(
            bucket=config.OBJECT_STORAGE_BUCKET,
            client_factory=functools.cache(lambda: _s3_client(config)),
            max_bytes=config.ASSET_FETCH_MAX_BYTES,
            timeout_seconds=config.ASSET_FETCH_TIMEOUT_SECONDS,
        )

    @asynccontextmanager
    async def fetch(
        self,
        *,
        tenant_id: str,
        asset_id: str,
        storage_uri: str,
        content_sha256: str,
        mime_type: str,
        file_size_bytes: int | None = None,
    ) -> AsyncIterator[FetchedAsset]:
        key = resolve_storage_reference(
            storage_uri,
            bucket=self._bucket,
            tenant_id=tenant_id,
            asset_id=asset_id,
            content_sha256=content_sha256,
        )
        expected_sha256 = content_sha256.lower()
        if file_size_bytes is not None and file_size_bytes > self._max_bytes:
            raise PermanentIngestionError(
                "ASSET_TOO_LARGE", f"the asset is over the {self._max_bytes}-byte fetch limit"
            )
        fd, path = tempfile.mkstemp(prefix="asset-", dir=self._temp_dir)
        stop = threading.Event()
        try:
            with os.fdopen(fd, "wb") as out:
                try:
                    async with asyncio.timeout(self._timeout_seconds):
                        size, sha256, head = await asyncio.to_thread(self._download, key, out, stop)
                except TimeoutError:
                    raise TransientIngestionError(
                        "ASSET_FETCH_TIMEOUT",
                        f"the asset was not fetched within {self._timeout_seconds:g} s",
                    ) from None
            if sha256 != expected_sha256:
                raise PermanentIngestionError(
                    "ASSET_INTEGRITY_MISMATCH", "the stored object does not match its SHA-256"
                )
            if file_size_bytes is not None and size != file_size_bytes:
                raise PermanentIngestionError(
                    "ASSET_INTEGRITY_MISMATCH", "the stored object does not match its size"
                )
            detected = detect_mime_type(head)
            if detected is None or detected != mime_type.lower():
                raise PermanentIngestionError(
                    "ASSET_TYPE_MISMATCH", "the stored object is not of its declared type"
                )
            yield FetchedAsset(path=path, size_bytes=size, sha256=sha256, mime_type=detected)
        except IngestionFailure as failure:
            logger.warning("asset fetch failed: %s (retryable=%s)", failure.code, failure.retryable)
            raise
        finally:
            stop.set()  # an abandoned download thread stops at its next chunk
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def _download(self, key: str, out: BinaryIO, stop: threading.Event) -> tuple[int, str, bytes]:
        """Stream one object to ``out``; runs in a worker thread."""
        try:
            response = self._client_factory().get_object(Bucket=self._bucket, Key=key)
        except IngestionFailure:
            raise
        except ClientError as exc:
            raise classify_client_error(exc) from None
        except NoCredentialsError:
            raise PermanentIngestionError(
                "OBJECT_STORAGE_NOT_CONFIGURED", "object storage credentials are not configured"
            ) from None
        except (
            EndpointConnectionError,
            ConnectTimeoutError,
            ReadTimeoutError,
            BotoConnectionError,
        ):
            raise TransientIngestionError(
                "OBJECT_STORAGE_UNAVAILABLE", "object storage is unreachable"
            ) from None
        except BotoCoreError as exc:
            raise PermanentIngestionError(
                "OBJECT_STORAGE_FAILURE", f"object storage client error ({type(exc).__name__})"
            ) from None
        body = response["Body"]
        try:
            declared = response.get("ContentLength")
            if isinstance(declared, int) and declared > self._max_bytes:
                raise PermanentIngestionError(
                    "ASSET_TOO_LARGE", f"the asset is over the {self._max_bytes}-byte fetch limit"
                )
            digest = hashlib.sha256()
            size = 0
            head = b""
            for chunk in body.iter_chunks(CHUNK_BYTES):
                if stop.is_set():
                    raise _Stopped
                size += len(chunk)
                if size > self._max_bytes:
                    raise PermanentIngestionError(
                        "ASSET_TOO_LARGE",
                        f"the asset is over the {self._max_bytes}-byte fetch limit",
                    )
                digest.update(chunk)
                out.write(chunk)
                if len(head) < _HEAD_BYTES:
                    head += chunk[: _HEAD_BYTES - len(head)]
            out.flush()
            return size, digest.hexdigest(), head
        except (ReadTimeoutError, ResponseStreamingError, BotoConnectionError):
            raise TransientIngestionError(
                "OBJECT_STORAGE_UNAVAILABLE", "the object stream was interrupted"
            ) from None
        finally:
            body.close()
