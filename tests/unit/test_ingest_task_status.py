"""Signed ingestion task status for reconciliation (M3-09, C23 at AI-service)."""

import os

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.replay_guard import RedisReplayGuard
from app.core.service_context import ServiceContext
from app.main import create_app
from app.services.tasks.task_records import (
    RedisTaskRecordStore,
    TaskRecord,
    get_task_record_store,
)
from tests.signing import TENANT_A, TENANT_B, signed_headers, user_context

REDIS_URL = os.getenv("TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL not set")

EVENT = {
    "event_id": "3f1c0d9e-6b0a-4c55-9d43-1c2b8f7a9e10",
    "job_id": "job-1",
    "external_task_id": "task_abc",
    "run_number": 1,
    "attempt_count": 2,
    "event_type": "job_failed",
    "error_code": "INGESTION_PIPELINE_UNAVAILABLE",
    "error_message": "no pipeline",
    "is_retryable": False,
}


@pytest.fixture
async def guard():
    guard = RedisReplayGuard.connect(REDIS_URL, timeout=2.0)
    yield guard
    await guard.close()


@pytest.fixture
async def stored(in_memory_task_records):
    await in_memory_task_records.create(
        TaskRecord(
            external_task_id="task_abc",
            tenant_id=TENANT_A,
            job_id="job-1",
            run_number=1,
            attempt_count=2,
            state="failed",
            events=[EVENT],
        )
    )
    return in_memory_task_records


def system(tenant_id: str = TENANT_A) -> ServiceContext:
    return ServiceContext.system(tenant_id=tenant_id, correlation_id="reconcile-1")


async def get(app, path, headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://ai-service") as http:
        return await http.get(path, headers=headers)


def signed_get(path: str, context: ServiceContext, **kwargs) -> dict[str, str]:
    return signed_headers(b"", context=context, target=path, method="GET", **kwargs)


PATH = "/v1/ingest/tasks/task_abc"


async def test_the_tasks_tenant_gets_its_status_and_events(guard, stored):
    app = create_app(replay_guard=guard)
    response = await get(app, PATH, signed_get(PATH, system()))
    assert response.status_code == 200, response.text
    data = response.json()
    assert (data["job_id"], data["run_number"], data["attempt_count"], data["state"]) == (
        "job-1",
        1,
        2,
        "failed",
    )
    assert data["events"][0]["event_id"] == EVENT["event_id"]
    assert data["events"][0]["error_code"] == "INGESTION_PIPELINE_UNAVAILABLE"


async def test_another_tenant_gets_404(guard, stored):
    app = create_app(replay_guard=guard)
    response = await get(app, PATH, signed_get(PATH, system(TENANT_B)))
    assert response.status_code == 404


async def test_an_unknown_task_gets_404(guard, stored):
    app = create_app(replay_guard=guard)
    path = "/v1/ingest/tasks/task_missing"
    assert (await get(app, path, signed_get(path, system()))).status_code == 404


async def test_a_user_context_is_refused_even_with_the_dispatch_permission(guard, stored):
    app = create_app(replay_guard=guard)
    response = await get(app, PATH, signed_get(PATH, user_context(TENANT_A)))
    assert response.status_code == 403


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer anything"}, {"X-School-ID": "1"}],
    ids=["none", "bearer", "school-header"],
)
async def test_unsigned_requests_are_401(guard, stored, headers):
    app = create_app(replay_guard=guard)
    assert (await get(app, PATH, headers)).status_code == 401


async def test_a_signature_for_another_task_path_is_401(guard, stored):
    app = create_app(replay_guard=guard)
    headers = signed_get("/v1/ingest/tasks/task_other", system())
    assert (await get(app, PATH, headers)).status_code == 401


async def test_record_store_outage_is_503(guard):
    app = create_app(replay_guard=guard)
    down = RedisTaskRecordStore.connect("redis://127.0.0.1:1/0", timeout=0.3)
    app.dependency_overrides[get_task_record_store] = lambda: down
    try:
        response = await get(app, PATH, signed_get(PATH, system()))
    finally:
        await down.close()
    assert response.status_code == 503


async def test_the_full_record_round_trips_through_real_redis(guard):
    store = RedisTaskRecordStore.connect(REDIS_URL, timeout=2.0)
    app = create_app(replay_guard=guard)
    app.dependency_overrides[get_task_record_store] = lambda: store
    try:
        await store.create(
            TaskRecord(
                external_task_id="task_real",
                tenant_id=TENANT_A,
                job_id="job-1",
                run_number=1,
                attempt_count=2,
                events=[{**EVENT, "external_task_id": "task_real"}],
            )
        )
        path = "/v1/ingest/tasks/task_real"
        response = await get(app, path, signed_get(path, system()))
    finally:
        await store.close()
    assert response.status_code == 200
    assert response.json()["events"][0]["external_task_id"] == "task_real"
