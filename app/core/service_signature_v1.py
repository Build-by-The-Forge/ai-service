"""BUILD Service Signature v1 — AI-service implementation.

Specification: build-backend ``docs/phase1-m3-boundary-design.md`` Part A.
The backend implements the same protocol separately; the shared test vector
(A.7), not shared code, proves that both sides compute identical signatures.

This module is framework-free. Request-body limits and replay protection
(A.5 steps 6 and 8) are layered on top of ``ServiceVerifier``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from app.core.service_context import (
    ServiceContext,
    ServiceContextError,
    parse_service_context,
)

PROTOCOL_VERSION = "v1"

# Sender ids, one per direction (A.1).
BACKEND_SERVICE_ID = "build-backend"  # backend -> AI-service, BACKEND_TO_AI_SECRET
AI_SERVICE_ID = "ai-service"  # AI-service -> backend, AI_TO_BACKEND_SECRET

SERVICE_HEADER = "X-BUILD-Service"
TIMESTAMP_HEADER = "X-BUILD-Timestamp"
REQUEST_ID_HEADER = "X-BUILD-Request-ID"
CONTEXT_HEADER = "X-BUILD-Context"
SIGNATURE_HEADER = "X-BUILD-Signature"
REQUIRED_HEADERS = (
    SERVICE_HEADER,
    TIMESTAMP_HEADER,
    REQUEST_ID_HEADER,
    CONTEXT_HEADER,
    SIGNATURE_HEADER,
)

MAX_CLOCK_SKEW_SECONDS = 60
CONTEXT_HEADER_MAX_LENGTH = 4096
SHA256_OF_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

_SIGNATURE_PATTERN = re.compile(r"v1=[0-9a-f]{64}")
_TIMESTAMP_PATTERN = re.compile(r"[1-9][0-9]{0,11}")
_UNPADDED_BASE64URL = re.compile(r"[A-Za-z0-9\-_]*")


class SignatureVerificationError(Exception):
    """Raised for any verification failure.

    ``detail`` is meant for internal logs. Every cause must produce the same
    401 response so callers learn nothing about which check failed (A.5).
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@dataclass(frozen=True)
class VerifiedCall:
    """What a receiver learns from a request that passed verification."""

    sender: str
    timestamp: int
    request_id: str
    context: ServiceContext
    raw_context: bytes


def hex_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def serialize_context(context: Mapping[str, Any]) -> bytes:
    """Canonical context bytes (A.4): sorted keys, compact, ASCII-escaped."""
    text = json.dumps(context, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return text.encode("ascii")


def context_to_header(context: Mapping[str, Any]) -> str:
    encoded = base64.urlsafe_b64encode(serialize_context(context)).decode("ascii")
    return encoded.rstrip("=")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate keys in context")
    return dict(pairs)


def _forbid_constant(token: str) -> Any:
    raise ValueError(f"{token} is not allowed in context")


def parse_context_header(value: str) -> tuple[bytes, dict[str, Any]]:
    """Return ``(raw_bytes, object)`` for a context header in canonical form.

    Only the encoding is checked here (unpadded base64url, a JSON object that
    re-serialises byte-identically); field rules are the context contract's.
    """
    if len(value) > CONTEXT_HEADER_MAX_LENGTH:
        raise SignatureVerificationError("context header exceeds 4096 characters")
    if _UNPADDED_BASE64URL.fullmatch(value) is None or len(value) % 4 == 1:
        raise SignatureVerificationError("context header is not unpadded base64url")
    padded = value + "=" * (-len(value) % 4)
    try:
        raw = base64.b64decode(padded, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SignatureVerificationError("context header failed to decode") from exc
    if context_bytes_to_header(raw) != value:
        raise SignatureVerificationError("context header has non-canonical base64url encoding")

    try:
        parsed = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_forbid_constant,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise SignatureVerificationError("context is not valid ASCII JSON") from exc
    if not isinstance(parsed, dict):
        raise SignatureVerificationError("context must be a JSON object")
    if serialize_context(parsed) != raw:
        raise SignatureVerificationError("context JSON is not canonical")
    return raw, parsed


def context_bytes_to_header(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def wire_request_target(raw_path: bytes, query_string: bytes) -> str:
    """Path and query exactly as received (ASGI ``raw_path``/``query_string``)."""
    try:
        target = raw_path.decode("ascii")
        query = query_string.decode("ascii")
    except UnicodeDecodeError as exc:
        raise SignatureVerificationError("request target contains non-ASCII bytes") from exc
    if query:
        target += "?" + query
    return target


@dataclass(frozen=True)
class CanonicalRequest:
    sender: str
    method: str
    target: str
    timestamp: str
    request_id: str
    body_sha256: str
    context_sha256: str

    def to_bytes(self) -> bytes:
        lines = [
            PROTOCOL_VERSION,
            self.sender,
            self.method.upper(),
            self.target,
            self.timestamp,
            self.request_id,
            self.body_sha256,
            self.context_sha256,
        ]
        return "\n".join(lines).encode("utf-8")

    def signature(self, secret: str) -> str:
        mac = hmac.new(secret.encode("utf-8"), self.to_bytes(), hashlib.sha256)
        return f"{PROTOCOL_VERSION}={mac.hexdigest()}"


@dataclass(frozen=True)
class ServiceSigner:
    """Signs outgoing requests for one direction with its current secret."""

    sender: str
    secret: str

    def headers(
        self,
        *,
        method: str,
        target: str,
        body: bytes,
        context: ServiceContext,
        timestamp: int,
        request_id: str | None = None,
    ) -> dict[str, str]:
        rid = request_id if request_id is not None else str(uuid.uuid4())
        raw_context = serialize_context(context.to_json_object())
        canonical = CanonicalRequest(
            sender=self.sender,
            method=method,
            target=target,
            timestamp=str(timestamp),
            request_id=rid,
            body_sha256=hex_sha256(body),
            context_sha256=hex_sha256(raw_context),
        )
        return {
            SERVICE_HEADER: self.sender,
            TIMESTAMP_HEADER: str(timestamp),
            REQUEST_ID_HEADER: rid,
            CONTEXT_HEADER: context_bytes_to_header(raw_context),
            SIGNATURE_HEADER: canonical.signature(self.secret),
        }


def _is_canonical_uuid4(value: str) -> bool:
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return False
    return parsed.version == 4 and str(parsed) == value


def _collect_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    lookup = {name.lower(): name for name in REQUIRED_HEADERS}
    collected: dict[str, str] = {}
    for name, value in headers:
        canonical_name = lookup.get(name.lower())
        if canonical_name is None:
            continue
        if canonical_name in collected:
            raise SignatureVerificationError(f"{canonical_name} sent more than once")
        collected[canonical_name] = value
    absent = [name for name in REQUIRED_HEADERS if name not in collected]
    if absent:
        raise SignatureVerificationError("absent headers: " + ", ".join(absent))
    return collected


@dataclass(frozen=True)
class PendingCall:
    """Headers that passed A.5 steps 1-5; the signature is still unchecked."""

    sender: str
    timestamp: str
    request_id: str
    signature: str
    context: ServiceContext
    raw_context: bytes


@dataclass(frozen=True)
class ServiceVerifier:
    """Verifies requests from the single sender this receiver accepts.

    ``secrets`` holds the direction's current secret first, then optionally
    the previous one during rotation.
    """

    accepted_sender: str
    secrets: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.secrets or not all(self.secrets):
            raise ValueError("ServiceVerifier needs at least one non-empty secret")

    def verify(
        self,
        *,
        method: str,
        target: str,
        headers: Iterable[tuple[str, str]],
        body_sha256: str,
        now: int,
    ) -> VerifiedCall:
        """A.5 steps 1-5 and 7 for a body whose digest is already known."""
        pending = self.precheck(headers=headers, now=now)
        return self.finish(pending, method=method, target=target, body_sha256=body_sha256)

    def precheck(self, *, headers: Iterable[tuple[str, str]], now: int) -> PendingCall:
        """A.5 steps 1-5, in order, before any of the body is read.

        ``headers`` is an iterable of (name, value) pairs, so repeated headers
        are visible.
        """
        values = _collect_headers(headers)

        signature = values[SIGNATURE_HEADER]
        if _SIGNATURE_PATTERN.fullmatch(signature) is None:
            raise SignatureVerificationError("signature header is malformed")

        sender = values[SERVICE_HEADER]
        if sender != self.accepted_sender:
            raise SignatureVerificationError("sender is not accepted by this receiver")

        timestamp = values[TIMESTAMP_HEADER]
        if _TIMESTAMP_PATTERN.fullmatch(timestamp) is None:
            raise SignatureVerificationError("timestamp is malformed")
        if abs(now - int(timestamp)) > MAX_CLOCK_SKEW_SECONDS:
            raise SignatureVerificationError("timestamp is outside the accepted window")

        request_id = values[REQUEST_ID_HEADER]
        if not _is_canonical_uuid4(request_id):
            raise SignatureVerificationError("request id is not a canonical UUIDv4")

        raw_context, decoded = parse_context_header(values[CONTEXT_HEADER])
        try:
            context = parse_service_context(decoded)
        except ServiceContextError as exc:
            raise SignatureVerificationError(f"context is invalid: {exc}") from exc

        return PendingCall(
            sender=sender,
            timestamp=timestamp,
            request_id=request_id,
            signature=signature,
            context=context,
            raw_context=raw_context,
        )

    def finish(
        self, pending: PendingCall, *, method: str, target: str, body_sha256: str
    ) -> VerifiedCall:
        """A.5 step 7: the signature, against each configured secret."""
        canonical = CanonicalRequest(
            sender=pending.sender,
            method=method,
            target=target,
            timestamp=pending.timestamp,
            request_id=pending.request_id,
            body_sha256=body_sha256,
            context_sha256=hex_sha256(pending.raw_context),
        )
        matched = False
        for secret in self.secrets:
            if hmac.compare_digest(canonical.signature(secret), pending.signature):
                matched = True
        if not matched:
            raise SignatureVerificationError("signature does not match")

        return VerifiedCall(
            sender=pending.sender,
            timestamp=int(pending.timestamp),
            request_id=pending.request_id,
            context=pending.context,
            raw_context=pending.raw_context,
        )
