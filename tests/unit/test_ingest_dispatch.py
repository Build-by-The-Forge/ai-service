"""Signed ingestion dispatch (M3-06; C2, C7, C12, C14, C15, C29 at AI-service).

The real app is used with a real Redis replay guard (TEST_REDIS_URL); the
fail-closed case points a real client at a closed port.
"""

import os

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.core.idempotency import InMemoryIdempotencyStore, set_idempotency_store
from app.core.replay_guard import RedisReplayGuard
from app.core.service_context import ServiceContext
from app.main import create_app
from app.services.tasks.dispatcher import get_task_dispatcher
from tests.signing import (
    DISPATCH_PATH,
    TENANT_A,
    TENANT_B,
    dispatch_body,
    signed_headers,
    user_context,
)

REDIS_URL = os.getenv("TEST_REDIS_URL")
real_redis = pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL not set")


class RecordingDispatcher:
    def __init__(self) -> None:
        self.tasks: list[dict] = []

    async def dispatch_ingest(self, **kwargs):
        self.tasks.append(kwargs)


@pytest.fixture(autouse=True)
def fresh_idempotency():
    set_idempotency_store(InMemoryIdempotencyStore())


@pytest.fixture
async def guard():
    guard = RedisReplayGuard.connect(REDIS_URL, timeout=2.0)
    yield guard
    await guard.close()


def build(guard):
    app = create_app(replay_guard=guard)
    recorder = RecordingDispatcher()
    app.dependency_overrides[get_task_dispatcher] = lambda: recorder
    return app, recorder


def client(app) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://ai-service")


async def post(app, body, headers, path=DISPATCH_PATH):
    async with client(app) as http:
        return await http.post(path, content=body, headers=headers)


# --- accepted ---------------------------------------------------------------------


@real_redis
async def test_signed_dispatch_is_accepted_for_the_context_tenant(guard):
    app, recorder = build(guard)
    body = dispatch_body()
    response = await post(app, body, signed_headers(body))

    assert response.status_code == 202, response.text
    data = response.json()
    assert data["status"] == "ACCEPTED"
    assert data["external_task_id"].startswith("task_")
    assert "tenant_id" not in data["data"]
    [task] = recorder.tasks
    assert task["tenant_id"] == TENANT_A  # from the signed context, as a string UUID
    assert task["external_task_id"] == data["external_task_id"]
    assert (task["job_id"], task["run_number"], task["attempt_count"]) == ("job-1", 1, 1)
    assert "tenant_id" not in task["payload"] and "callback_url" not in task["payload"]


@real_redis
async def test_same_idempotency_key_is_a_duplicate_within_a_tenant(guard):
    app, recorder = build(guard)
    body = dispatch_body()
    first = await post(app, body, signed_headers(body))
    second = await post(app, body, signed_headers(body))  # new request id, same key
    assert first.status_code == second.status_code == 202
    assert second.json()["status"] == "DUPLICATE"
    assert second.json()["external_task_id"] == first.json()["external_task_id"]
    assert len(recorder.tasks) == 1


@real_redis
async def test_idempotency_is_scoped_by_the_context_tenant(guard):
    app, recorder = build(guard)
    body = dispatch_body()
    a = await post(app, body, signed_headers(body, context=user_context(TENANT_A)))
    b = await post(app, body, signed_headers(body, context=user_context(TENANT_B)))
    assert a.json()["status"] == b.json()["status"] == "ACCEPTED"
    assert a.json()["external_task_id"] != b.json()["external_task_id"]
    assert [t["tenant_id"] for t in recorder.tasks] == [TENANT_A, TENANT_B]


# --- the body cannot name identity or a destination (C15, C18, C29) ----------------


@real_redis
@pytest.mark.parametrize(
    "extra",
    [
        {"tenant_id": TENANT_B},
        {"user_id": "0b9d8c7a-6e5f-4a3b-8c2d-000000000000"},
        {"membership_id": "x"},
        {"role": "owner"},
        {"callback_url": "https://attacker.example/hook"},
        {"metadata": {"tenant_id": TENANT_B}},
        {"authority": "REGULATORY"},
    ],
)
async def test_identity_or_destination_fields_in_the_body_are_rejected(guard, extra):
    app, recorder = build(guard)
    body = dispatch_body(**extra)
    response = await post(app, body, signed_headers(body))
    assert response.status_code == 422
    assert recorder.tasks == []


@real_redis
async def test_invalid_body_is_422(guard):
    app, recorder = build(guard)
    body = dispatch_body(content_sha256="not-a-hash")
    assert (await post(app, body, signed_headers(body))).status_code == 422
    assert recorder.tasks == []


# --- permission: the context only narrows (C14) -------------------------------------


@real_redis
@pytest.mark.parametrize(
    "context",
    [
        user_context(permissions=("ai.use",)),
        user_context(permissions=("ai.questions.generate", "content.read"), role="owner"),
        user_context(permissions=(), role="admin"),
        ServiceContext.system(tenant_id=TENANT_A, correlation_id="sweep-1"),
    ],
    ids=["other-permission", "owner-without-it", "admin-role-only", "system"],
)
async def test_context_without_the_dispatch_permission_is_403(guard, context):
    app, recorder = build(guard)
    body = dispatch_body(content_sha256="not-a-hash")  # 403 wins over body validation
    response = await post(app, body, signed_headers(body, context=context))
    assert response.status_code == 403
    assert recorder.tasks == []


# --- authentication (C2, C7, C12) ---------------------------------------------------


@real_redis
async def test_tampered_body_is_401(guard):
    app, recorder = build(guard)
    headers = signed_headers(dispatch_body())
    response = await post(app, dispatch_body(job_id="job-other"), headers)
    assert (response.status_code, response.json()) == (
        401,
        {"error": "invalid service authentication"},
    )
    assert recorder.tasks == []


@real_redis
async def test_request_signed_with_the_ai_to_backend_secret_is_401(guard):
    app, recorder = build(guard)
    body = dispatch_body()
    response = await post(app, body, signed_headers(body, secret=settings.AI_TO_BACKEND_SECRET))
    assert response.status_code == 401
    assert recorder.tasks == []


@real_redis
async def test_request_claiming_to_be_ai_service_is_401(guard):
    app, recorder = build(guard)
    body = dispatch_body()
    response = await post(app, body, signed_headers(body, sender="ai-service"))
    assert response.status_code == 401
    assert recorder.tasks == []


@real_redis
@pytest.mark.parametrize(
    "headers",
    [
        {"Authorization": "Bearer build-ai-dev-secret-key"},
        {"Authorization": f"Bearer {settings.BACKEND_TO_AI_SECRET}"},
        {"X-Service-Identity": "build-backend", "X-Timestamp": "1", "X-Service-Signature": "ab"},
        {"X-School-ID": "1"},
        {},
    ],
    ids=["static-bearer", "secret-as-bearer", "pre-m3-scheme", "school-header", "none"],
)
async def test_unsigned_requests_are_401(guard, headers):
    app, recorder = build(guard)
    response = await post(app, dispatch_body(), {**headers, "Content-Type": "application/json"})
    assert response.status_code == 401
    assert recorder.tasks == []


@real_redis
async def test_replayed_dispatch_is_401(guard):
    app, recorder = build(guard)
    body = dispatch_body()
    headers = signed_headers(body)
    assert (await post(app, body, headers)).status_code == 202
    assert (await post(app, body, headers)).status_code == 401
    assert len(recorder.tasks) == 1


@real_redis
async def test_the_duplicate_api_v1_mount_is_gone(guard):
    app, recorder = build(guard)
    body = dispatch_body()
    path = "/api/v1/ingest/dispatch"
    response = await post(app, body, signed_headers(body, target=path), path=path)
    assert response.status_code == 404
    assert recorder.tasks == []


# --- fail closed --------------------------------------------------------------------


async def test_unreachable_redis_is_503_and_nothing_is_dispatched():
    dead = RedisReplayGuard.connect("redis://127.0.0.1:1/0", timeout=0.5)
    try:
        app, recorder = build(dead)
        body = dispatch_body()
        response = await post(app, body, signed_headers(body))
        assert response.status_code == 503
        assert recorder.tasks == []
    finally:
        await dead.close()


async def test_without_an_inbound_secret_every_dispatch_is_refused(monkeypatch):
    monkeypatch.setattr(settings, "BACKEND_TO_AI_SECRET", None)
    dead = RedisReplayGuard.connect("redis://127.0.0.1:1/0", timeout=0.5)
    try:
        app, recorder = build(dead)
        body = dispatch_body()
        response = await post(app, body, signed_headers(body, secret="x" * 40))
        assert response.status_code == 401
        assert recorder.tasks == []
    finally:
        await dead.close()


# --- M3-09: the dispatch records the task for reconciliation -------------------------


@real_redis
async def test_an_accepted_dispatch_records_the_task(guard, in_memory_task_records):
    app, recorder = build(guard)
    body = dispatch_body()
    data = (await post(app, body, signed_headers(body))).json()
    stored = await in_memory_task_records.get(data["external_task_id"])
    assert (
        stored.tenant_id,
        stored.job_id,
        stored.run_number,
        stored.attempt_count,
        stored.state,
    ) == (
        TENANT_A,
        "job-1",
        1,
        1,
        "accepted",
    )
    # A duplicate dispatch reuses the task; no second record appears.
    await post(app, body, signed_headers(body))
    assert len(in_memory_task_records.records) == 1


@real_redis
async def test_dispatch_is_refused_when_task_records_are_unavailable(guard):
    from app.services.tasks.task_records import (
        RedisTaskRecordStore,
        get_task_record_store,
    )

    app, recorder = build(guard)
    down = RedisTaskRecordStore.connect("redis://127.0.0.1:1/0", timeout=0.3)
    app.dependency_overrides[get_task_record_store] = lambda: down
    try:
        body = dispatch_body()
        response = await post(app, body, signed_headers(body))
    finally:
        await down.close()
    assert response.status_code == 503
    assert recorder.tasks == []


class UnavailableIdempotency:
    async def set_if_absent(self, **kwargs):
        from app.core.idempotency import IdempotencyUnavailableError

        raise IdempotencyUnavailableError("down")


@real_redis
async def test_dispatch_fails_closed_when_idempotency_is_unavailable(guard):
    set_idempotency_store(UnavailableIdempotency())
    app, recorder = build(guard)
    body = dispatch_body()
    response = await post(app, body, signed_headers(body))
    assert response.status_code == 503
    assert recorder.tasks == []
