"""Signed AI routes under /v1/ai (M3-10; C12, C13, C14, C29 at AI-service).

The real app with a real Redis replay guard (TEST_REDIS_URL). The model is
the deterministic provider; memory is a recorder that reads the request scope
at call time, so the tests show which tenant and user a route acted for.
"""

import json
import os

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.replay_guard import RedisReplayGuard
from app.core.tenant import get_tenant_id
from app.main import create_app
from app.services.ai.deterministic import DeterministicLLM
from app.services.ai.dependencies import get_ai_orchestrator
from app.services.ai.orchestrator import AIOrchestrator
from app.services.ai.tenanted_retriever import TenantAwareRetrievalService
from tests.signing import TENANT_A, TENANT_B, USER_ID, signed_headers, user_context

REDIS_URL = os.getenv("TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL not set")

ROOT = {
    "id": "c1",
    "title": "Course",
    "level": 0,
    "children": [{"id": "t1", "title": "T", "level": 1}],
}

# path, permission, a valid body
ROUTES = [
    ("/v1/ai/chat", "ai.use", {"message": "What is osmosis?"}),
    ("/v1/ai/explain", "ai.use", {"subject": "osmosis"}),
    ("/v1/ai/generate-questions", "ai.questions.generate", {"topic": "cells", "count": 3}),
    ("/v1/ai/retrieve", "ai.use", {"query": "osmosis"}),
    ("/v1/ai/tree/generate", "ai.use", {"course_title": "Biology", "topics": ["Cells"]}),
    ("/v1/ai/read/start", "ai.use", {"root": ROOT}),
]
ROUTE_IDS = [r[0] for r in ROUTES]


class RecordingMemory:
    def __init__(self) -> None:
        self.reads: list[tuple[str, str]] = []

    async def get_user_memory(self, user_id: str):
        self.reads.append((get_tenant_id(), user_id))
        return [{"topic": f"memory of {user_id}", "mastery_score": 0.5}]


@pytest.fixture
async def guard():
    guard = RedisReplayGuard.connect(REDIS_URL, timeout=2.0)
    yield guard
    await guard.close()


@pytest.fixture
def memory():
    return RecordingMemory()


@pytest.fixture
def app(guard, memory):
    app = create_app(replay_guard=guard)
    app.dependency_overrides[get_ai_orchestrator] = lambda: AIOrchestrator(
        llm=DeterministicLLM(), retriever=TenantAwareRetrievalService(), memory=memory
    )
    return app


async def post(app, path: str, body: bytes, headers: dict[str, str] | None = None):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://ai") as http:
        return await http.post(path, content=body, headers=headers or {})


def encode(body: dict) -> bytes:
    return json.dumps(body).encode()


def signed(path: str, body: bytes, *permissions: str, tenant_id: str = TENANT_A):
    return signed_headers(
        body, target=path, context=user_context(tenant_id=tenant_id, permissions=permissions)
    )


# --- accepted, scoped to the context ----------------------------------------------------


@pytest.mark.parametrize("path,permission,body", ROUTES, ids=ROUTE_IDS)
async def test_a_signed_user_call_with_the_permission_is_served(app, path, permission, body):
    raw = encode(body)
    response = await post(app, path, raw, signed(path, raw, permission))
    assert response.status_code == 200, response.text
    assert response.json()["success"] is True


async def index_osmosis_notes(engine) -> None:
    """Each tenant has its own passage; retrieval must return the caller's."""
    for tenant_id in (TENANT_A, TENANT_B):
        await engine.index_document(
            tenant_id=tenant_id,
            document_id=f"notes-{tenant_id}",
            document_title="Biology notes",
            text=f"Tenant {tenant_id} notes: osmosis moves water across a membrane.",
        )


async def test_chat_uses_the_context_tenant_and_user(app, memory, in_memory_retrieval):
    await index_osmosis_notes(in_memory_retrieval)
    raw = encode({"message": "What is osmosis?"})
    response = await post(app, "/v1/ai/chat", raw, signed("/v1/ai/chat", raw, "ai.use"))

    data = response.json()["data"]
    assert memory.reads == [(TENANT_A, USER_ID)]
    assert data["memory_used"] == f"memory of {USER_ID}: mastery=0.5"
    [context] = data["context_used"]
    assert context["text"].startswith(f"Tenant {TENANT_A} notes")
    assert data["answer"].startswith("[deterministic model]")


async def test_retrieval_follows_the_context_tenant(app, in_memory_retrieval):
    await index_osmosis_notes(in_memory_retrieval)
    raw = encode({"query": "osmosis"})
    for tenant_id in (TENANT_A, TENANT_B):
        headers = signed("/v1/ai/retrieve", raw, "ai.use", tenant_id=tenant_id)
        response = await post(app, "/v1/ai/retrieve", raw, headers)
        [hit] = response.json()["data"]
        assert hit["text"].startswith(f"Tenant {tenant_id} notes")


# --- C29: the body cannot name an identity ---------------------------------------------


@pytest.mark.parametrize("field", ["tenant_id", "user_id", "membership_id", "role"])
@pytest.mark.parametrize("path,permission,body", ROUTES, ids=ROUTE_IDS)
async def test_a_body_naming_an_identity_is_422(app, path, permission, body, field):
    raw = encode({**body, field: TENANT_B if field == "tenant_id" else "someone-else"})
    response = await post(app, path, raw, signed(path, raw, permission))
    assert response.status_code == 422


# --- C14: the context narrows only -----------------------------------------------------


@pytest.mark.parametrize("path,permission,body", ROUTES, ids=ROUTE_IDS)
async def test_a_context_without_the_route_permission_is_403(app, path, permission, body):
    other = "ai.use" if permission == "ai.questions.generate" else "ai.questions.generate"
    raw = encode(body)
    response = await post(app, path, raw, signed(path, raw, other, "content.job.manage"))
    assert response.status_code == 403


async def test_a_system_context_is_403(app):
    from app.core.service_context import ServiceContext

    raw = encode({"message": "hi"})
    context = ServiceContext.system(tenant_id=TENANT_A, correlation_id="corr")
    headers = signed_headers(raw, target="/v1/ai/chat", context=context)
    assert (await post(app, "/v1/ai/chat", raw, headers)).status_code == 403


# --- C12, C13: closed to clients -------------------------------------------------------


@pytest.mark.parametrize("path,permission,body", ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2ln"},
        {"Authorization": "Bearer build-ai-dev-secret-key"},
        {"X-School-ID": "1"},
        {"X-School-ID": "1", "X-Tenant-ID": TENANT_A, "X-User-ID": USER_ID},
    ],
    ids=["no-auth", "end-user-jwt", "static-bearer", "x-school-id", "identity-headers"],
)
async def test_unsigned_calls_are_401(app, path, permission, body, headers):
    raw = encode(body)
    response = await post(app, path, raw, {**headers, "Content-Type": "application/json"})
    assert response.status_code == 401


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/ai/chat",
        "/api/v1/ai/explain",
        "/api/v1/ai/generate-questions",
        "/api/v1/ai/retrieve/",
        "/api/v1/ai/tree/generate",
        "/api/v1/ai/read/start",
    ],
)
async def test_the_old_unsigned_paths_are_gone(app, path):
    assert (await post(app, path, b"{}", {"X-School-ID": "1"})).status_code in (404, 405)


async def test_a_body_changed_after_signing_is_401(app):
    raw = encode({"message": "What is osmosis?"})
    headers = signed("/v1/ai/chat", raw, "ai.use")
    tampered = encode({"message": "What is osmosiz?"})
    assert (await post(app, "/v1/ai/chat", tampered, headers)).status_code == 401


# --- input limits ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,permission,body",
    [
        ("/v1/ai/chat", "ai.use", {"message": "x" * 4001}),
        ("/v1/ai/chat", "ai.use", {"message": ""}),
        ("/v1/ai/explain", "ai.use", {"subject": "x" * 4001}),
        ("/v1/ai/generate-questions", "ai.questions.generate", {"topic": "t", "count": 51}),
        ("/v1/ai/generate-questions", "ai.questions.generate", {"topic": "t", "count": 0}),
        ("/v1/ai/retrieve", "ai.use", {"query": "x" * 4001}),
    ],
)
async def test_inputs_beyond_the_limits_are_422(app, path, permission, body):
    raw = encode(body)
    assert (await post(app, path, raw, signed(path, raw, permission))).status_code == 422


async def test_the_limits_themselves_are_accepted(app):
    raw = encode({"topic": "t", "count": 50})
    path = "/v1/ai/generate-questions"
    response = await post(app, path, raw, signed(path, raw, "ai.questions.generate"))
    assert response.status_code == 200


# --- M5: structured question generation -------------------------------------------------


async def test_generated_questions_are_structured(app, in_memory_retrieval):
    await index_osmosis_notes(in_memory_retrieval)  # questions need course material
    raw = encode(
        {"topic": "cells", "count": 3, "difficulty": "hard", "question_type": "true_false"}
    )
    path = "/v1/ai/generate-questions"
    response = await post(app, path, raw, signed(path, raw, "ai.questions.generate"))
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["discarded"] == 0
    assert len(data["questions"]) == 3
    for question in data["questions"]:
        assert question["question_type"] == "true_false"
        assert question["difficulty"] == "hard"
        assert sorted(o["text"] for o in question["options"]) == ["False", "True"]
        assert sum(o["is_correct"] for o in question["options"]) == 1


@pytest.mark.parametrize("change", [{"difficulty": "extreme"}, {"question_type": "essay"}])
async def test_unknown_difficulty_or_type_is_422(app, change):
    raw = encode({"topic": "cells", **change})
    path = "/v1/ai/generate-questions"
    response = await post(app, path, raw, signed(path, raw, "ai.questions.generate"))
    assert response.status_code == 422


# --- M6: correlation id -------------------------------------------------------------------


async def test_the_verified_correlation_id_is_echoed_and_logged(app, caplog):
    import logging

    from app.core.logging import JsonFormatter

    raw = encode({"message": "hello"})
    response = await post(app, "/v1/ai/chat", raw, signed("/v1/ai/chat", raw, "ai.use"))
    assert response.headers["x-correlation-id"] == "corr-test"

    # A log line written while serving carries it too.
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "inside", None, None)
    from app.core.correlation import current_correlation_id

    token = current_correlation_id.set("corr-test")
    try:
        assert '"correlation_id": "corr-test"' in JsonFormatter().format(record)
    finally:
        current_correlation_id.reset(token)
    assert '"correlation_id"' not in JsonFormatter().format(record)


async def test_rejected_requests_carry_no_correlation_header(app):
    raw = encode({"message": "hello"})
    response = await post(app, "/v1/ai/chat", raw, {"Content-Type": "application/json"})
    assert response.status_code == 401
    assert "x-correlation-id" not in response.headers
