import abc
import asyncio
import logging
import random
import time
from urllib.parse import urlparse

import httpx

from app.core.callbacks.events import CallbackDeliveryResult, CallbackEvent
from app.core.callbacks.signer import CallbackSigner, canonical_serialize
from app.core.config import settings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}


class CallbackPort(abc.ABC):
    """
    Abstract Port for delivering domain lifecycle callback events to external consumers.
    """

    @abc.abstractmethod
    async def dispatch(self, callback_url: str, event: CallbackEvent) -> CallbackDeliveryResult:
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

    async def dispatch(self, callback_url: str, event: CallbackEvent) -> CallbackDeliveryResult:
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

        self._events.append(
            {
                "callback_url": callback_url,
                "event": event,
                "timestamp": time.time(),
            }
        )

        return CallbackDeliveryResult(
            success=True,
            status_code=200,
            attempts=1,
            retryable=False,
            delivered_at=time.time(),
            event_id=event.event_id,
        )


class HTTPCallbackDispatcher(CallbackPort):
    """
    Production-grade HTTP callback dispatcher with HMAC signing, canonical serialization,
    and bounded exponential backoff with jitter.
    """

    def __init__(
        self,
        signer: CallbackSigner | None = None,
        timeout_seconds: float | None = None,
        max_attempts: int | None = None,
        initial_backoff: float | None = None,
        max_backoff: float | None = None,
        jitter: bool | None = None,
    ):
        self._signer = signer or CallbackSigner(settings.SERVICE_AUTH_SECRET)
        self._timeout = timeout_seconds or settings.CALLBACK_TIMEOUT_SECONDS
        self._max_attempts = max_attempts or settings.CALLBACK_MAX_ATTEMPTS
        self._initial_backoff = initial_backoff or settings.CALLBACK_INITIAL_BACKOFF_SECONDS
        self._max_backoff = max_backoff or settings.CALLBACK_MAX_BACKOFF_SECONDS
        self._jitter = jitter if jitter is not None else settings.CALLBACK_JITTER

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                f"Invalid callback URL scheme: '{parsed.scheme}'. Must be http or https."
            )
        if not parsed.netloc:
            raise ValueError(f"Invalid callback URL host in '{url}'.")

    async def dispatch(self, callback_url: str, event: CallbackEvent) -> CallbackDeliveryResult:
        try:
            self._validate_url(callback_url)
        except ValueError as err:
            logger.error("Callback URL validation failed: %s", str(err))
            return CallbackDeliveryResult(
                success=False,
                attempts=1,
                retryable=False,
                error_code="INVALID_CALLBACK_URL",
                error_message=str(err),
                event_id=event.event_id,
            )

        payload_bytes = canonical_serialize(event)
        timestamp = int(time.time())
        signature = self._signer.sign(payload_bytes, timestamp)

        headers = {
            "Content-Type": "application/json",
            "X-Build-Signature": signature,
            "X-Timestamp": str(timestamp),
            "X-Correlation-ID": event.correlation_id,
            "X-Event-ID": event.event_id,
            "X-Tenant-ID": event.tenant_id,
        }

        last_error_code = None
        last_error_message = None
        last_status_code = None
        is_retryable = False

        for attempt in range(1, self._max_attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        callback_url,
                        content=payload_bytes,
                        headers=headers,
                    )
                    last_status_code = response.status_code

                    # 2xx Success
                    if 200 <= response.status_code < 300:
                        logger.info(
                            "Callback delivered successfully: event=%s type=%s url=%s status=%d attempt=%d",
                            event.event_id,
                            event.event_type,
                            callback_url,
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

                    # Non-retryable client error (400, 401, 403, 404, 422)
                    if response.status_code in NON_RETRYABLE_STATUS_CODES:
                        last_error_code = f"HTTP_{response.status_code}"
                        last_error_message = f"Server rejected callback with status {response.status_code}: {response.text[:200]}"
                        is_retryable = False
                        logger.warning(
                            "Non-retryable callback rejection: event=%s url=%s status=%d",
                            event.event_id,
                            callback_url,
                            response.status_code,
                        )
                        break

                    # Retryable server error or rate limit
                    last_error_code = f"HTTP_{response.status_code}"
                    last_error_message = f"Server returned retryable status {response.status_code}"
                    is_retryable = True

            except (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError) as exc:
                last_error_code = type(exc).__name__
                last_error_message = str(exc)
                is_retryable = True
                logger.warning(
                    "Transient network failure delivering callback: event=%s url=%s attempt=%d/%d error=%s",
                    event.event_id,
                    callback_url,
                    attempt,
                    self._max_attempts,
                    last_error_message,
                )

            except Exception as unhandled:
                last_error_code = "UNHANDLED_ERROR"
                last_error_message = str(unhandled)
                is_retryable = False
                logger.error(
                    "Unhandled error delivering callback: %s", str(unhandled), exc_info=True
                )
                break

            # Calculate exponential backoff if more attempts remain
            if attempt < self._max_attempts and is_retryable:
                backoff = min(self._initial_backoff * (2 ** (attempt - 1)), self._max_backoff)
                if self._jitter:
                    backoff += random.uniform(0, 0.1 * backoff)
                await asyncio.sleep(backoff)

        # Exhausted attempts or permanent failure
        logger.error(
            "Callback delivery failed permanently: event=%s type=%s url=%s status=%s error=%s",
            event.event_id,
            event.event_type,
            callback_url,
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
