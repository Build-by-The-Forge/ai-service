"""Inbound verification of BUILD Service Signature v1 requests (design §A.5).

Pure ASGI, so it runs before FastAPI parses the body: the 1 MiB limit is
applied while the body arrives, never after buffering all of it. For every
path the predicate selects, the order is:

    headers / sender / timestamp / request id / context   -> 401
    body, hashed chunk by chunk, at most 1 MiB             -> 413
    signature (current secret, then previous)              -> 401
    first use of the request id in Redis                   -> 401 on reuse,
                                                              503 if Redis is down

The request id is only recorded after the signature verifies, so a forger
cannot burn ids. Rejections never reach the route, and every 401 carries the
same body; the specific reason goes to the log only.

On success the route sees the original body and ``request.state.service_call``
(a ``VerifiedCall``).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from typing import Protocol

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.correlation import current_correlation_id
from app.core.replay_guard import ReplayGuardUnavailableError
from app.core.service_signature_v1 import (
    ServiceVerifier,
    SignatureVerificationError,
    wire_request_target,
)

logger = logging.getLogger(__name__)

BODY_LIMIT_BYTES = 1_048_576
VERIFIED_CALL_STATE = "service_call"

_REJECTIONS = {
    401: b'{"error":"invalid service authentication"}',
    413: b'{"error":"request body too large"}',
    503: b'{"error":"service temporarily unavailable"}',
}


class ReplayGuard(Protocol):
    async def first_use(self, receiver: str, sender: str, request_id: str) -> bool:
        ...


class _Rejected(Exception):
    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


class _Disconnected(Exception):
    pass


class SignedServiceRequestMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        receiver_id: str,
        verifier: ServiceVerifier | None,
        replay_guard: ReplayGuard,
        applies_to: Callable[[str], bool],
        body_limit: int = BODY_LIMIT_BYTES,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.app = app
        self.receiver_id = receiver_id
        self.verifier = verifier
        self.replay_guard = replay_guard
        self.applies_to = applies_to
        self.body_limit = body_limit
        self.now = now

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.applies_to(scope["path"]):
            await self.app(scope, receive, send)
            return
        try:
            body, call = await self._authenticate(scope, receive)
        except _Rejected as rejected:
            logger.warning(
                "signed service request rejected: status=%s reason=%s path=%s",
                rejected.status,
                rejected.reason,
                scope["path"],
            )
            await _send_rejection(send, rejected.status)
            return
        except _Disconnected:
            return

        scope.setdefault("state", {})[VERIFIED_CALL_STATE] = call
        correlation_id = call.context.correlation_id
        token = current_correlation_id.set(correlation_id)

        async def send_with_correlation(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-correlation-id", correlation_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, _body_then(receive, body), send_with_correlation)
        finally:
            current_correlation_id.reset(token)

    async def _authenticate(self, scope: Scope, receive: Receive):
        if self.verifier is None:
            # No inbound secret configured (development only): nothing verifies.
            raise _Rejected(401, "service secret not configured")
        header_pairs = [
            (name.decode("latin-1"), value.decode("latin-1")) for name, value in scope["headers"]
        ]
        try:
            pending = self.verifier.precheck(headers=header_pairs, now=int(self.now()))
            target = wire_request_target(
                scope.get("raw_path") or scope["path"].encode("utf-8"),
                scope.get("query_string") or b"",
            )
        except SignatureVerificationError as exc:
            raise _Rejected(401, exc.detail) from exc

        body, digest = await self._consume_body(header_pairs, receive)

        try:
            call = self.verifier.finish(
                pending, method=scope["method"], target=target, body_sha256=digest
            )
        except SignatureVerificationError as exc:
            raise _Rejected(401, exc.detail) from exc

        try:
            fresh = await self.replay_guard.first_use(
                self.receiver_id, call.sender, call.request_id
            )
        except ReplayGuardUnavailableError as exc:
            raise _Rejected(503, f"replay guard unavailable ({exc})") from exc
        if not fresh:
            raise _Rejected(401, "request id already used")
        return body, call

    async def _consume_body(
        self, header_pairs: list[tuple[str, str]], receive: Receive
    ) -> tuple[bytes, str]:
        for name, value in header_pairs:
            if name.lower() == "content-length" and value.isdigit():
                if int(value) > self.body_limit:
                    raise _Rejected(413, f"declared length {value} exceeds limit")
        digest = hashlib.sha256()
        received = bytearray()
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                raise _Disconnected
            chunk = message.get("body", b"")
            if len(received) + len(chunk) > self.body_limit:
                raise _Rejected(413, "streamed body exceeds limit")
            digest.update(chunk)
            received.extend(chunk)
            more = message.get("more_body", False)
        return bytes(received), digest.hexdigest()


async def _send_rejection(send: Send, status: int) -> None:
    payload = _REJECTIONS[status]
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})


def _body_then(receive: Receive, body: bytes) -> Receive:
    delivered = False

    async def wrapped() -> Message:
        nonlocal delivered
        if delivered:
            return await receive()
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    return wrapped
