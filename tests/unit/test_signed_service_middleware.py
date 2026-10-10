"""Inbound signed-request handling with a real Redis (C8, C9, C10, C11).

Set TEST_REDIS_URL, e.g. redis://127.0.0.1:6379/15. The fail-closed cases
need no server: they use a real client aimed at a port nobody listens on.
"""

import os
import time
import uuid

import httpx
import pytest
import redis.asyncio as redis_asyncio
from fastapi import FastAPI, Request
from pydantic import BaseModel

from app.core.middleware.service_auth import (
    BODY_LIMIT_BYTES,
    VERIFIED_CALL_STATE,
    SignedServiceRequestMiddleware,
)
from app.core.replay_guard import REPLAY_WINDOW_SECONDS, RedisReplayGuard, replay_key
from app.core.service_context import parse_service_context
from app.core.service_signature_v1 import (
    AI_SERVICE_ID,
    BACKEND_SERVICE_ID,
    CONTEXT_HEADER,
    REQUEST_ID_HEADER,
    SERVICE_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    ServiceSigner,
    ServiceVerifier,
)

REDIS_URL = os.getenv("TEST_REDIS_URL")
UNREACHABLE_REDIS = "redis://127.0.0.1:1/0"
real_redis = pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL not set")

SECRET = "test-only-backend-to-ai-secret-0123456789abcdef"
ROUTE = "/v1/ingest/dispatch"
TENANT = "6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c"
CONTEXT = parse_service_context(
    {
        "v": 1,
        "actor": "system",
        "tenant_id": TENANT,
        "user_id": None,
        "membership_id": None,
        "role": None,
        "permissions": [],
        "correlation_id": "job-9:attempt-1",
    }
)
INVALID_AUTH = {"error": "invalid service authentication"}

signer = ServiceSigner(sender=BACKEND_SERVICE_ID, secret=SECRET)


class Dispatch(BaseModel):
    job_id: str


def make_app(guard):
    handled = []
    app = FastAPI()

    @app.post(ROUTE)
    async def dispatch(payload: Dispatch, request: Request):
        call = getattr(request.state, VERIFIED_CALL_STATE)
        handled.append(payload.job_id)
        return {"job_id": payload.job_id, "tenant": call.context.tenant_id}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.add_middleware(
        SignedServiceRequestMiddleware,
        receiver_id=AI_SERVICE_ID,
        verifier=ServiceVerifier(accepted_sender=BACKEND_SERVICE_ID, secrets=(SECRET,)),
        replay_guard=guard,
        applies_to=lambda path: path.startswith("/v1/ingest/"),
    )
    return app, handled


def body_for(job_id="job-9"):
    return f'{{"job_id":"{job_id}"}}'.encode()


def sign(body, *, at=None, request_id=None, target=ROUTE):
    headers = signer.headers(
        method="POST",
        target=target,
        body=body,
        context=CONTEXT,
        timestamp=int(time.time()) if at is None else at,
        request_id=request_id,
    )
    headers["content-type"] = "application/json"  # not part of the signature
    return headers


def http(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ai-service")


@pytest.fixture
async def guard():
    guard = RedisReplayGuard.connect(REDIS_URL, timeout=2.0)
    yield guard
    await guard.close()


@pytest.fixture
async def down_guard():
    guard = RedisReplayGuard.connect(UNREACHABLE_REDIS, timeout=0.5)
    yield guard
    await guard.close()


async def drive(app, headers, chunks):
    """Call the ASGI app directly with a chunked body (no Content-Length) and
    report the status plus how many bytes the app pulled from the client."""
    queue = list(chunks)
    pulled = []
    out = []

    async def receive():
        if not queue:
            return {"type": "http.disconnect"}
        chunk = queue.pop(0)
        pulled.append(len(chunk))
        return {"type": "http.request", "body": chunk, "more_body": bool(queue)}

    async def send(message):
        out.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": ROUTE,
            "raw_path": ROUTE.encode(),
            "query_string": b"",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "client": ("127.0.0.1", 9),
            "server": ("ai-service", 8000),
        },
        receive,
        send,
    )
    status = [m["status"] for m in out if m["type"] == "http.response.start"][0]
    return status, pulled


# ------------------------------------------------------------------- valid


@real_redis
async def test_valid_request_is_handled_once_with_context(guard):
    app, handled = make_app(guard)
    body = body_for()
    async with http(app) as client:
        response = await client.post(ROUTE, content=body, headers=sign(body))
    assert response.status_code == 200, response.text
    assert response.json() == {"job_id": "job-9", "tenant": TENANT}
    assert handled == ["job-9"]


@real_redis
async def test_other_paths_pass_through(guard):
    app, _ = make_app(guard)
    async with http(app) as client:
        assert (await client.get("/health")).json() == {"status": "ok"}


@real_redis
async def test_a_body_of_exactly_one_mebibyte_is_accepted(guard):
    app = FastAPI()
    lengths = []

    @app.post(ROUTE)
    async def raw(request: Request):
        lengths.append(len(await request.body()))
        return {}

    app.add_middleware(
        SignedServiceRequestMiddleware,
        receiver_id=AI_SERVICE_ID,
        verifier=ServiceVerifier(accepted_sender=BACKEND_SERVICE_ID, secrets=(SECRET,)),
        replay_guard=guard,
        applies_to=lambda path: path == ROUTE,
    )
    body = b"a" * BODY_LIMIT_BYTES
    async with http(app) as client:
        assert (await client.post(ROUTE, content=body, headers=sign(body))).status_code == 200
    assert lengths == [BODY_LIMIT_BYTES]


@real_redis
async def test_streamed_body_within_limit_reaches_the_route(guard):
    app, handled = make_app(guard)
    body = body_for("job-chunked")
    status, pulled = await drive(app, sign(body), [body[:5], body[5:12], body[12:]])
    assert status == 200
    assert sum(pulled) == len(body)
    assert handled == ["job-chunked"]


# -------------------------------------------------------------- C8: stale


@real_redis
@pytest.mark.parametrize("skew", [-61, 61, -86400])
async def test_requests_outside_sixty_seconds_are_refused(guard, skew):
    app, handled = make_app(guard)
    body = body_for()
    async with http(app) as client:
        response = await client.post(
            ROUTE, content=body, headers=sign(body, at=int(time.time()) + skew)
        )
    assert (response.status_code, response.json()) == (401, INVALID_AUTH)
    assert handled == []


# ------------------------------------------------------------- C9: replay


@real_redis
async def test_replaying_a_request_is_refused(guard):
    app, handled = make_app(guard)
    body = body_for()
    headers = sign(body)
    async with http(app) as client:
        first = await client.post(ROUTE, content=body, headers=headers)
        again = await client.post(ROUTE, content=body, headers=headers)
    assert first.status_code == 200
    assert (again.status_code, again.json()) == (401, INVALID_AUTH)
    assert handled == ["job-9"]

    client = redis_asyncio.Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        key = replay_key(AI_SERVICE_ID, BACKEND_SERVICE_ID, headers[REQUEST_ID_HEADER])
        assert await client.get(key) == "1"
        assert 0 < await client.ttl(key) <= REPLAY_WINDOW_SECONDS
    finally:
        await client.close()


@real_redis
async def test_request_id_reuse_with_a_fresh_signature_is_refused(guard):
    app, handled = make_app(guard)
    rid = str(uuid.uuid4())
    async with http(app) as client:
        await client.post(ROUTE, content=body_for("a"), headers=sign(body_for("a"), request_id=rid))
        second = await client.post(
            ROUTE, content=body_for("b"), headers=sign(body_for("b"), request_id=rid)
        )
    assert second.status_code == 401
    assert handled == ["a"]


@real_redis
async def test_forged_request_does_not_consume_the_request_id(guard):
    app, handled = make_app(guard)
    body = body_for()
    headers = sign(body)
    async with http(app) as client:
        forged = await client.post(ROUTE, content=body_for("evil"), headers=headers)
        genuine = await client.post(ROUTE, content=body, headers=headers)
    assert (forged.status_code, genuine.status_code) == (401, 200)
    assert handled == ["job-9"]


@real_redis
async def test_replay_keys_are_separate_per_receiver(guard):
    rid = str(uuid.uuid4())
    assert await guard.first_use("build-backend", AI_SERVICE_ID, rid)
    assert await guard.first_use(AI_SERVICE_ID, BACKEND_SERVICE_ID, rid)
    assert not await guard.first_use(AI_SERVICE_ID, BACKEND_SERVICE_ID, rid)


# ---------------------------------------------------------- C10: malformed


@real_redis
@pytest.mark.parametrize(
    "header,value",
    [
        (SIGNATURE_HEADER, None),
        (TIMESTAMP_HEADER, None),
        (CONTEXT_HEADER, "e30"),  # "{}" — fails the context schema
        (CONTEXT_HEADER, "not base64!"),
        (REQUEST_ID_HEADER, "123"),
        (TIMESTAMP_HEADER, "yesterday"),
        (SERVICE_HEADER, AI_SERVICE_ID),
        (SIGNATURE_HEADER, "Bearer old-static-key"),
    ],
)
async def test_malformed_requests_are_refused_uniformly(guard, header, value):
    app, handled = make_app(guard)
    body = body_for()
    headers = sign(body)
    if value is None:
        del headers[header]
    else:
        headers[header] = value
    async with http(app) as client:
        response = await client.post(ROUTE, content=body, headers=headers)
    assert (response.status_code, response.json()) == (401, INVALID_AUTH)
    assert handled == []


@real_redis
async def test_bearer_only_request_is_refused(guard):
    app, handled = make_app(guard)
    async with http(app) as client:
        response = await client.post(
            ROUTE, content=body_for(), headers={"authorization": "Bearer " + SECRET}
        )
    assert (response.status_code, response.json()) == (401, INVALID_AUTH)
    assert handled == []


@real_redis
async def test_unsigned_query_parameters_are_refused(guard):
    app, handled = make_app(guard)
    body = body_for()
    async with http(app) as client:
        response = await client.post(ROUTE + "?tenant=other", content=body, headers=sign(body))
    assert response.status_code == 401
    assert handled == []


@real_redis
async def test_malformed_headers_are_refused_without_reading_the_body(guard):
    app, handled = make_app(guard)
    headers = sign(b"irrelevant")
    headers[TIMESTAMP_HEADER] = "0"
    status, pulled = await drive(app, headers, [b"z" * 4096] * 8)
    assert status == 401
    assert pulled == []
    assert handled == []


# ---------------------------------------------------------- oversized body


@real_redis
async def test_declared_length_over_the_limit_is_refused(guard):
    app, handled = make_app(guard)
    body = b"a" * (BODY_LIMIT_BYTES + 1)
    async with http(app) as client:
        response = await client.post(ROUTE, content=body, headers=sign(body))
    assert response.status_code == 413
    assert handled == []


@real_redis
async def test_streamed_body_is_cut_off_once_it_passes_the_limit(guard):
    app, handled = make_app(guard)
    piece = b"a" * (128 * 1024)
    pieces = [piece] * 32  # 4 MiB offered without Content-Length
    status, pulled = await drive(app, sign(b"".join(pieces)), pieces)
    assert status == 413
    assert sum(pulled) <= BODY_LIMIT_BYTES + len(piece)
    assert len(pulled) < len(pieces)
    assert handled == []


# -------------------------------------------------------- C11: fail closed


async def test_unreachable_redis_answers_503_and_runs_nothing(down_guard):
    app, handled = make_app(down_guard)
    body = body_for()
    async with http(app) as client:
        response = await client.post(ROUTE, content=body, headers=sign(body))
    assert (response.status_code, response.json()) == (
        503,
        {"error": "service temporarily unavailable"},
    )
    assert handled == []


async def test_bad_signature_is_401_even_when_redis_is_down(down_guard):
    app, handled = make_app(down_guard)
    async with http(app) as client:
        response = await client.post(ROUTE, content=body_for("x"), headers=sign(body_for("y")))
    assert response.status_code == 401
    assert handled == []


async def test_a_replay_guard_that_never_answers_times_out_closed():
    import asyncio

    async def silent(reader, writer):
        await asyncio.sleep(30)

    server = await asyncio.start_server(silent, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    stuck = RedisReplayGuard.connect(f"redis://127.0.0.1:{port}/0", timeout=0.2)
    try:
        app, handled = make_app(stuck)
        body = body_for()
        started = time.monotonic()
        async with http(app) as client:
            response = await client.post(ROUTE, content=body, headers=sign(body))
        elapsed = time.monotonic() - started
    finally:
        await stuck.close()
        server.close()
    assert response.status_code == 503
    assert handled == []
    assert elapsed < 3
