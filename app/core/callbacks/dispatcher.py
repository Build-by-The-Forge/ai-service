"""Ingestion callbacks from AI-service to the backend (design B.3).

A callback is a signed BUILD Service Signature v1 request:

* destination: ``BACKEND_CALLBACK_URL`` only (validated operator
  configuration); nothing in a task, payload or event can change it;
* redirects are never followed: a 3xx is a delivery failure;
* context: a ``system`` context naming the task's tenant (recorded from the
  verified dispatch), with no user and no permissions;
* secret: ``AI_TO_BACKEND_SECRET`` (current only; previous secrets verify,
  they never sign);
* body: the backend's event vocabulary, with no tenant or user field;
* every attempt is signed afresh, so a retry is never a replay.
"""

import abc
import asyncio
import json
import logging
import random
import re
import time
from typing import Any

import httpx

from app.core.callbacks.events import CallbackDeliveryResult, CallbackEvent, CallbackEventType
from app.core.config import settings
from app.core.service_context import ServiceContext, ServiceContextError
from app.core.service_signature_v1 import AI_SERVICE_ID, ServiceSigner
from app.schemas.ingest import IngestCallbackPayload

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 409, 413, 422}

_CORRELATION_ID = re.compile(r"[A-Za-z0-9._:\-]{1,128}")


def backend_callback_body(event: CallbackEvent) -> dict[str, Any] | None:
    """Translate a worker lifecycle event into the backend's callback body.

    STAGE_COMPLETED -> stage_progress (with its evidence), COMPLETED ->
    job_completed, FAILED -> job_failed. Returns None for events the backend
    has no transition for (STARTED, PROGRESS): the job is already
    ``processing`` once the dispatch was accepted.
    """
    if event.event_type == CallbackEventType.STAGE_COMPLETED:
        body: dict[str, Any] = {
            "event_type": "stage_progress",
            "stage": event.stage,
            "metrics": dict(event.metrics or {}),
        }
    elif event.event_type == CallbackEventType.COMPLETED:
        metrics: dict[str, Any] = dict(event.metrics or {})
        if event.extracted_metadata is not None:
            metrics["extracted_metadata"] = event.extracted_metadata
        if event.chunk_summary is not None:
            metrics["chunk_summary"] = event.chunk_summary
        body = {"event_type": "job_completed", "metrics": metrics}
    elif event.event_type == CallbackEventType.FAILED:
        body = {
            "event_type": "job_failed",
            "error_code": event.error_code or "AI_PROCESSING_FAILED",
            "error_message": event.error_message or "AI-service reported job failure",
            "is_retryable": bool(event.retryable),
        }
    else:
        return None

    body.update(
        event_id=event.event_id,
        job_id=event.job_id,
        external_task_id=event.external_task_id,
        run_number=event.run_number,
        attempt_count=event.attempt_count,
    )
    return IngestCallbackPayload(**body).model_dump(exclude_unset=True)


def _context_correlation_id(event: CallbackEvent) -> str:
    for candidate in (event.correlation_id, event.external_task_id, event.job_id):
        if candidate and _CORRELATION_ID.fullmatch(candidate):
            return candidate
    return "ai-callback"


class CallbackPort(abc.ABC):
    """Delivers lifecycle events to the backend. The destination is never a
    parameter: it is fixed by configuration."""

    @abc.abstractmethod
    async def dispatch(self, event: CallbackEvent) -> CallbackDeliveryResult:
        pass


class InMemoryCallbackDispatcher(CallbackPort):
    """
    In-memory callback dispatcher for tests and local debugging.
    """

    def __init__(self):
        self._events: list[dict] = []
        self._should_fail: bool = False
        self._failure_status: int = 500

    def simulate_failure(self, status_code: int = 500) -> None:
        self._should_fail = True
        self._failure_status = status_code

    def reset(self) -> None:
        self._events.clear()
        self._should_fail = False

    @property
    def delivered_events(self) -> list[CallbackEvent]:
        return [item["event"] for item in self._events]

    @property
    def deliveries(self) -> list[dict]:
        return self._events

    async def dispatch(self, event: CallbackEvent) -> CallbackDeliveryResult:
        if self._should_fail:
            retryable = self._failure_status in RETRYABLE_STATUS_CODES
            return CallbackDeliveryResult(
                success=False,
                status_code=self._failure_status,
                attempts=1,
                retryable=retryable,
                error_code="SIMULATED_FAILURE",
                error_message=f"Simulated HTTP {self._failure_status} error",
                event_id=event.event_id,
            )

        self._events.append({"event": event, "timestamp": time.time()})

        return CallbackDeliveryResult(
            success=True,
            status_code=200,
            attempts=1,
            retryable=False,
            delivered_at=time.time(),
            event_id=event.event_id,
        )


class HTTPCallbackDispatcher(CallbackPort):
    """Signed HTTP delivery to BACKEND_CALLBACK_URL with bounded exponential
    backoff and jitter."""

    def __init__(
        self,
        timeout_seconds: float | None = None,
        max_attempts: int | None = None,
        initial_backoff: float | None = None,
        max_backoff: float | None = None,
        jitter: bool | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not settings.AI_TO_BACKEND_SECRET:
            raise RuntimeError("AI_TO_BACKEND_SECRET is not configured")
        if not settings.BACKEND_CALLBACK_URL:
            raise RuntimeError("BACKEND_CALLBACK_URL is not configured")
        self._signer = ServiceSigner(sender=AI_SERVICE_ID, secret=settings.AI_TO_BACKEND_SECRET)
        self._url = httpx.URL(settings.BACKEND_CALLBACK_URL)
        self._target = self._url.raw_path.decode("ascii")  # signed exactly as sent
        self._transport = transport
        self._timeout = timeout_seconds or settings.CALLBACK_TIMEOUT_SECONDS
        self._max_attempts = max_attempts or settings.CALLBACK_MAX_ATTEMPTS
        self._initial_backoff = initial_backoff or settings.CALLBACK_INITIAL_BACKOFF_SECONDS
        self._max_backoff = max_backoff or settings.CALLBACK_MAX_BACKOFF_SECONDS
        self._jitter = jitter if jitter is not None else settings.CALLBACK_JITTER

    def _failure(self, event: CallbackEvent, code: str, message: str) -> CallbackDeliveryResult:
        logger.error("Callback not sent: event=%s code=%s %s", event.event_id, code, message)
        return CallbackDeliveryResult(
            success=False,
            attempts=0,
            retryable=False,
            error_code=code,
            error_message=message,
            event_id=event.event_id,
        )

    async def dispatch(self, event: CallbackEvent) -> CallbackDeliveryResult:
        body = backend_callback_body(event)
        if body is None:
            logger.debug(
                "Lifecycle event %s (%s) has no backend transition; not reported",
                event.event_id,
                event.event_type,
            )
            return CallbackDeliveryResult(success=True, attempts=0, event_id=event.event_id)

        try:
            context = ServiceContext.system(
                tenant_id=event.tenant_id, correlation_id=_context_correlation_id(event)
            )
        except ServiceContextError as exc:
            return self._failure(event, "INVALID_CALLBACK_CONTEXT", str(exc))

        payload_bytes = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")

        last_error_code = None
        last_error_message = None
        last_status_code = None
        is_retryable = False
        attempt = 0

        for attempt in range(1, self._max_attempts + 1):
            # A fresh request id and timestamp per attempt: the backend's
            # replay check would refuse a resent signed request.
            headers = self._signer.headers(
                method="POST",
                target=self._target,
                body=payload_bytes,
                context=context,
                timestamp=int(time.time()),
            )
            headers["Content-Type"] = "application/json"
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout, transport=self._transport, follow_redirects=False
                ) as client:
                    response = await client.post(self._url, content=payload_bytes, headers=headers)
                last_status_code = response.status_code

                if 200 <= response.status_code < 300:
                    logger.info(
                        "Callback delivered: event=%s type=%s status=%d attempt=%d",
                        event.event_id,
                        body["event_type"],
                        response.status_code,
                        attempt,
                    )
                    return CallbackDeliveryResult(
                        success=True,
                        status_code=response.status_code,
                        attempts=attempt,
                        retryable=False,
                        delivered_at=time.time(),
                        event_id=event.event_id,
                    )

                if 300 <= response.status_code < 400:
                    # Never followed: the configured URL is the only destination.
                    last_error_code = "CALLBACK_REDIRECT_REFUSED"
                    last_error_message = (
                        f"Backend answered {response.status_code}; redirects are not followed"
                    )
                    is_retryable = False
                    logger.warning(
                        "Callback redirect refused: event=%s status=%d",
                        event.event_id,
                        response.status_code,
                    )
                    break

                if response.status_code in NON_RETRYABLE_STATUS_CODES:
                    last_error_code = f"HTTP_{response.status_code}"
                    last_error_message = (
                        f"Backend rejected callback with status {response.status_code}"
                    )
                    is_retryable = False
                    logger.warning(
                        "Non-retryable callback rejection: event=%s status=%d",
                        event.event_id,
                        response.status_code,
                    )
                    break

                last_error_code = f"HTTP_{response.status_code}"
                last_error_message = f"Backend returned retryable status {response.status_code}"
                is_retryable = True

            except (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError) as exc:
                last_error_code = type(exc).__name__
                last_error_message = str(exc)
                is_retryable = True
                logger.warning(
                    "Transient failure delivering callback: event=%s attempt=%d/%d error=%s",
                    event.event_id,
                    attempt,
                    self._max_attempts,
                    last_error_message,
                )

            except Exception as unhandled:
                last_error_code = "UNHANDLED_ERROR"
                last_error_message = str(unhandled)
                is_retryable = False
                logger.error("Unhandled error delivering callback: %s", unhandled, exc_info=True)
                break

            if attempt < self._max_attempts and is_retryable:
                backoff = min(self._initial_backoff * (2 ** (attempt - 1)), self._max_backoff)
                if self._jitter:
                    backoff += random.uniform(0, 0.1 * backoff)
                await asyncio.sleep(backoff)

        logger.error(
            "Callback delivery failed: event=%s type=%s status=%s error=%s",
            event.event_id,
            body["event_type"],
            last_status_code,
            last_error_message,
        )
        return CallbackDeliveryResult(
            success=False,
            status_code=last_status_code,
            attempts=attempt,
            retryable=is_retryable,
            error_code=last_error_code,
            error_message=last_error_message,
            event_id=event.event_id,
        )


_default_callback_dispatcher: CallbackPort | None = None


def get_callback_dispatcher() -> CallbackPort:
    global _default_callback_dispatcher
    if _default_callback_dispatcher is None:
        _default_callback_dispatcher = HTTPCallbackDispatcher()
    return _default_callback_dispatcher


def set_callback_dispatcher(dispatcher: CallbackPort) -> None:
    global _default_callback_dispatcher
    _default_callback_dispatcher = dispatcher
