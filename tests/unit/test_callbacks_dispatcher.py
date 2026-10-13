"""Signed AI-service -> backend callbacks (M3-07, design B.3)."""

import json
import time

import httpx
import pytest

from app.core.callbacks.dispatcher import (
    HTTPCallbackDispatcher,
    InMemoryCallbackDispatcher,
    backend_callback_body,
)
from app.core.callbacks.events import CallbackEvent, CallbackEventType
from app.core.config import settings
from app.core.service_context import ServiceContext
from app.core.service_signature_v1 import (
    AI_SERVICE_ID,
    BACKEND_SERVICE_ID,
    ServiceVerifier,
    SignatureVerificationError,
    hex_sha256,
)

TENANT = "6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c"
CALLBACK_URL = "https://api.build.example/content/ingestion/callbacks"
A2B = "test-ai-to-backend-secret-for-callback-tests-0001"
B2A = "test-backend-to-ai-secret-for-callback-tests-0002"
BODY_FIELDS = {
    "event_id",
    "job_id",
    "external_task_id",
    "run_number",
    "attempt_count",
    "event_type",
    "metrics",
}


@pytest.fixture(autouse=True)
def callback_settings(monkeypatch):
    monkeypatch.setattr(settings, "BACKEND_CALLBACK_URL", CALLBACK_URL)
    monkeypatch.setattr(settings, "AI_TO_BACKEND_SECRET", A2B)
    monkeypatch.setattr(settings, "BACKEND_TO_AI_SECRET", B2A)


def event(event_type=CallbackEventType.COMPLETED, tenant_id=TENANT, **extra) -> CallbackEvent:
    return CallbackEvent(
        job_id="cjob-1",
        asset_id="asset-1",
        tenant_id=tenant_id,
        run_number=1,
        attempt_count=2,
        external_task_id="task_abc",
        correlation_id="corr-1",
        event_type=event_type,
        **extra,
    )


class Recorder:
    def __init__(self, *responses: httpx.Response) -> None:
        self.requests: list[httpx.Request] = []
        self._responses = list(responses) or [httpx.Response(200, json={"data": {}})]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


def dispatcher(recorder: Recorder, **kwargs) -> HTTPCallbackDispatcher:
    return HTTPCallbackDispatcher(
        timeout_seconds=5.0,
        max_attempts=kwargs.pop("max_attempts", 3),
        initial_backoff=0.001,
        max_backoff=0.002,
        jitter=False,
        transport=httpx.MockTransport(recorder),
    )


def verify(request: httpx.Request, secret: str = A2B, sender: str = AI_SERVICE_ID):
    return ServiceVerifier(accepted_sender=sender, secrets=(secret,)).verify(
        method=request.method,
        target=request.url.raw_path.decode(),
        headers=request.headers.multi_items(),
        body_sha256=hex_sha256(request.content),
        now=int(time.time()),
    )


# --- what is sent ---------------------------------------------------------------


async def test_completed_event_is_a_signed_job_completed_callback():
    recorder = Recorder()
    result = await dispatcher(recorder).dispatch(
        event(extracted_metadata={"pages": 3}, chunk_summary={"chunks": 9})
    )

    assert result.success is True
    [request] = recorder.requests
    assert str(request.url) == CALLBACK_URL
    assert "authorization" not in request.headers  # no Bearer, ever

    call = verify(request)
    assert call.context == ServiceContext.system(tenant_id=TENANT, correlation_id="corr-1")
    assert call.context.permission_list == ()  # a system context grants nothing

    body = json.loads(request.content)
    assert set(body) == BODY_FIELDS
    assert body["event_type"] == "job_completed"
    assert body["metrics"] == {"extracted_metadata": {"pages": 3}, "chunk_summary": {"chunks": 9}}
    assert (body["job_id"], body["external_task_id"]) == ("cjob-1", "task_abc")
    assert (body["run_number"], body["attempt_count"]) == (1, 2)


async def test_failed_event_is_a_job_failed_callback():
    recorder = Recorder()
    await dispatcher(recorder).dispatch(
        event(
            CallbackEventType.FAILED,
            error_code="OCR_TIMEOUT",
            error_message="took too long",
            retryable=True,
        )
    )
    body = json.loads(recorder.requests[0].content)
    assert body["event_type"] == "job_failed"
    assert (body["error_code"], body["error_message"], body["is_retryable"]) == (
        "OCR_TIMEOUT",
        "took too long",
        True,
    )


@pytest.mark.parametrize("event_type", [CallbackEventType.STARTED, CallbackEventType.PROGRESS])
async def test_events_without_a_backend_transition_are_not_sent(event_type):
    recorder = Recorder()
    result = await dispatcher(recorder).dispatch(event(event_type, progress_pct=10))
    assert (result.success, result.attempts) == (True, 0)
    assert recorder.requests == []


async def test_body_carries_no_identity_or_destination():
    recorder = Recorder()
    await dispatcher(recorder).dispatch(event())
    body = json.loads(recorder.requests[0].content)
    for forbidden in ("tenant_id", "user_id", "membership_id", "role", "callback_url", "asset_id"):
        assert forbidden not in body


async def test_callback_is_not_valid_under_the_backend_to_ai_secret_or_sender():
    recorder = Recorder()
    await dispatcher(recorder).dispatch(event())
    [request] = recorder.requests
    with pytest.raises(SignatureVerificationError, match="does not match"):
        verify(request, secret=B2A)
    with pytest.raises(SignatureVerificationError, match="sender"):
        verify(request, sender=BACKEND_SERVICE_ID)


async def test_tampering_with_the_sent_callback_breaks_the_signature():
    recorder = Recorder()
    await dispatcher(recorder).dispatch(event())
    [request] = recorder.requests
    tampered = httpx.Request(
        "POST",
        request.url,
        headers=request.headers,
        content=request.content.replace(b"job_completed", b"job_failed\x20\x20\x20"),
    )
    with pytest.raises(SignatureVerificationError, match="does not match"):
        verify(tampered)


async def test_a_task_without_a_uuid_tenant_is_not_sent():
    recorder = Recorder()
    result = await dispatcher(recorder).dispatch(event(tenant_id="school_futa"))
    assert (result.success, result.error_code) == (False, "INVALID_CALLBACK_CONTEXT")
    assert recorder.requests == []


async def test_correlation_id_outside_the_context_format_falls_back():
    recorder = Recorder()
    ev = CallbackEvent(
        job_id="cjob-1",
        asset_id="asset-1",
        tenant_id=TENANT,
        run_number=1,
        attempt_count=1,
        external_task_id="task_abc",
        correlation_id="has spaces/slashes",
        event_type=CallbackEventType.COMPLETED,
    )
    await dispatcher(recorder).dispatch(ev)
    assert verify(recorder.requests[0]).context.correlation_id == "task_abc"


# --- destination ------------------------------------------------------------------


async def test_only_the_configured_url_is_used():
    recorder = Recorder()
    await dispatcher(recorder).dispatch(event())
    assert [str(r.url) for r in recorder.requests] == [CALLBACK_URL]


def test_dispatch_takes_no_destination():
    import inspect

    params = list(inspect.signature(HTTPCallbackDispatcher.dispatch).parameters)
    assert params == ["self", "event"]


@pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
async def test_redirects_are_refused_not_followed(status_code):
    recorder = Recorder(
        httpx.Response(status_code, headers={"Location": "https://attacker.example/steal"})
    )
    result = await dispatcher(recorder).dispatch(event())
    assert result.success is False
    assert (result.error_code, result.retryable, result.attempts) == (
        "CALLBACK_REDIRECT_REFUSED",
        False,
        1,
    )
    assert [r.url.host for r in recorder.requests] == ["api.build.example"]


def test_missing_configuration_refuses_to_build_a_dispatcher(monkeypatch):
    monkeypatch.setattr(settings, "AI_TO_BACKEND_SECRET", None)
    with pytest.raises(RuntimeError, match="AI_TO_BACKEND_SECRET"):
        HTTPCallbackDispatcher()


# --- retries -----------------------------------------------------------------------


async def test_each_retry_is_signed_afresh():
    recorder = Recorder(
        httpx.Response(503, text="redis down"),
        httpx.Response(502),
        httpx.Response(200, json={}),
    )
    result = await dispatcher(recorder).dispatch(event())
    assert (result.success, result.attempts) == (True, 3)

    request_ids = [r.headers["x-build-request-id"] for r in recorder.requests]
    assert len(set(request_ids)) == 3  # a reused id would be refused as a replay
    bodies = {r.content for r in recorder.requests}
    assert len(bodies) == 1  # same event, same event_id
    for request in recorder.requests:
        verify(request)


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 409, 413, 422])
async def test_client_errors_are_not_retried(status_code):
    recorder = Recorder(httpx.Response(status_code))
    result = await dispatcher(recorder).dispatch(event())
    assert (result.success, result.retryable, result.attempts) == (False, False, 1)
    assert result.error_code == f"HTTP_{status_code}"


async def test_retries_are_bounded():
    recorder = Recorder(httpx.Response(503))
    result = await dispatcher(recorder, max_attempts=2).dispatch(event())
    assert (result.success, result.retryable, result.attempts) == (False, True, 2)
    assert len(recorder.requests) == 2


async def test_network_errors_are_retried():
    calls = 0

    def flaky(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={})

    d = HTTPCallbackDispatcher(
        max_attempts=3, initial_backoff=0.001, jitter=False, transport=httpx.MockTransport(flaky)
    )
    result = await d.dispatch(event())
    assert (result.success, result.attempts) == (True, 2)


# --- body helper and in-memory port -------------------------------------------------


def test_event_ids_are_uuid4():
    import uuid

    assert uuid.UUID(event().event_id).version == 4


def test_body_helper_matches_what_is_sent():
    ev = event()
    assert backend_callback_body(ev)["event_id"] == ev.event_id


async def test_in_memory_dispatcher_records_events():
    memory = InMemoryCallbackDispatcher()
    result = await memory.dispatch(event())
    assert result.success is True
    assert memory.delivered_events[0].event_type == CallbackEventType.COMPLETED


async def test_stage_events_become_signed_stage_progress_with_evidence():
    recorder = Recorder()
    evidence = {"characters": 900, "pages": 3, "method": "ocr"}
    await dispatcher(recorder).dispatch(
        event(CallbackEventType.STAGE_COMPLETED, stage="extracted", metrics=evidence)
    )
    [request] = recorder.requests
    verify(request)
    body = json.loads(request.content)
    assert (body["event_type"], body["stage"], body["metrics"]) == (
        "stage_progress",
        "extracted",
        evidence,
    )
    assert (body["run_number"], body["attempt_count"]) == (1, 2)
