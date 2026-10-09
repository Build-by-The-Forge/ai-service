"""Callback redirect safety over real sockets (M3-07, design B.3).

Two real HTTP servers run in this process: the configured backend answers
307 pointing at a second server. The signed callback must not be resent
there, and the delivery must be reported as a failure.
"""

import asyncio
import socket
import threading
import time

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Route

from app.core.callbacks.dispatcher import HTTPCallbackDispatcher
from app.core.callbacks.events import CallbackEvent, CallbackEventType
from app.core.config import settings

TENANT = "6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c"
PATH = "/content/ingestion/callbacks"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Server:
    def __init__(self, app: Starlette) -> None:
        self.port = _free_port()
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> "_Server":
        self._thread.start()
        deadline = time.time() + 10
        while not self._server.started:
            if time.time() > deadline:
                raise RuntimeError("test server did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *exc) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _event() -> CallbackEvent:
    return CallbackEvent(
        job_id="cjob-redirect",
        asset_id="asset-1",
        tenant_id=TENANT,
        run_number=1,
        attempt_count=1,
        external_task_id="task_redirect",
        correlation_id="corr-redirect",
        event_type=CallbackEventType.COMPLETED,
    )


@pytest.mark.parametrize("status_code", [302, 307, 308])
async def test_a_redirecting_backend_never_leads_the_callback_elsewhere(monkeypatch, status_code):
    elsewhere_hits: list[dict] = []
    backend_hits: list[str] = []

    async def elsewhere(request: Request):
        elsewhere_hits.append(dict(request.headers))
        return JSONResponse({"stolen": True})

    async def backend(request: Request):
        backend_hits.append(request.headers.get("x-build-request-id", ""))
        return RedirectResponse(redirect_target + PATH, status_code=status_code)

    with _Server(Starlette(routes=[Route(PATH, elsewhere, methods=["POST"])])) as sink:
        redirect_target = sink.base
        with _Server(Starlette(routes=[Route(PATH, backend, methods=["POST"])])) as origin:
            monkeypatch.setattr(settings, "BACKEND_CALLBACK_URL", origin.base + PATH)
            monkeypatch.setattr(settings, "AI_TO_BACKEND_SECRET", "a2b-" + "r" * 40)
            dispatcher = HTTPCallbackDispatcher(max_attempts=3, initial_backoff=0.01, jitter=False)
            result = await dispatcher.dispatch(_event())
            await asyncio.sleep(0.1)  # give any stray follow-up time to arrive

    assert result.success is False
    assert result.error_code == "CALLBACK_REDIRECT_REFUSED"
    assert result.status_code == status_code
    assert len(backend_hits) == 1  # refused, not retried
    assert elsewhere_hits == []  # the redirect target received nothing


async def test_a_real_backend_receives_the_signed_callback(monkeypatch):
    received: list[dict] = []

    async def backend(request: Request):
        received.append({"headers": dict(request.headers), "body": await request.body()})
        return JSONResponse({"data": {"status": "accepted"}})

    with _Server(Starlette(routes=[Route(PATH, backend, methods=["POST"])])) as origin:
        monkeypatch.setattr(settings, "BACKEND_CALLBACK_URL", origin.base + PATH)
        monkeypatch.setattr(settings, "AI_TO_BACKEND_SECRET", "a2b-" + "r" * 40)
        result = await HTTPCallbackDispatcher(max_attempts=1).dispatch(_event())

    assert result.success is True
    [hit] = received
    assert hit["headers"]["x-build-service"] == "ai-service"
    assert "authorization" not in hit["headers"]
