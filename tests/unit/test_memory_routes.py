"""Signed learning-memory routes (M5 design §3, M5-5).

Real app and real Redis replay guard; the store is a recorder that reads the
request scope, so the tests show whose memory each call touched.
"""

import json
import os
import types

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.ai.memory import get_memory_service
from app.core.replay_guard import RedisReplayGuard
from app.core.tenant import get_tenant_id
from app.main import create_app
from tests.signing import TENANT_A, USER_ID, signed_headers, user_context

REDIS_URL = os.getenv("TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL not set")

QUERY, UPSERT = "/v1/ai/memory/query", "/v1/ai/memory/upsert"


class RecordingMemory:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def _row(self, topic):
        return types.SimpleNamespace(topic=topic, mastery_score=0.6, attempts=1)

    async def query(self, user_id, topics=None):
        self.calls.append(("query", get_tenant_id(), user_id, topics))
        return [self._row(t) for t in topics or ["algebra"]]

    async def record(self, user_id, observations):
        self.calls.append(("record", get_tenant_id(), user_id, observations))
        return [self._row(t) for t, _ in observations]


@pytest.fixture
async def app():
    guard = RedisReplayGuard.connect(REDIS_URL, timeout=2.0)
    memory = RecordingMemory()
    app = create_app(replay_guard=guard)
    app.dependency_overrides[get_memory_service] = lambda: memory
    app.state.memory = memory
    yield app
    await guard.close()


async def post(app, path, body: dict, *permissions):
    raw = json.dumps(body).encode()
    headers = signed_headers(raw, target=path, context=user_context(permissions=permissions))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://ai") as http:
        return await http.post(path, content=raw, headers=headers)


async def test_query_reads_the_signed_users_memory(app):
    response = await post(app, QUERY, {"topics": ["algebra"]}, "ai.use")
    assert response.status_code == 200, response.text
    assert response.json()["data"]["items"] == [
        {"topic": "algebra", "mastery_score": 0.6, "attempts": 1}
    ]
    assert app.state.memory.calls == [("query", TENANT_A, USER_ID, ["algebra"])]


async def test_upsert_records_for_the_signed_user(app):
    body = {
        "observations": [
            {"topic": "algebra", "correct": True},
            {"topic": "cells", "correct": False},
        ]
    }
    response = await post(app, UPSERT, body, "assessment.attempt.submit")
    assert response.status_code == 200, response.text
    assert app.state.memory.calls == [
        ("record", TENANT_A, USER_ID, [("algebra", True), ("cells", False)])
    ]


@pytest.mark.parametrize(
    "path,body,permission",
    [
        (QUERY, {}, "assessment.attempt.submit"),
        (UPSERT, {"observations": [{"topic": "x", "correct": True}]}, "ai.use"),
    ],
)
async def test_each_route_needs_its_own_permission(app, path, body, permission):
    assert (await post(app, path, body, permission)).status_code == 403
    assert app.state.memory.calls == []


@pytest.mark.parametrize("field", ["tenant_id", "user_id", "membership_id", "role"])
@pytest.mark.parametrize(
    "path,body,permission",
    [
        (QUERY, {}, "ai.use"),
        (UPSERT, {"observations": [{"topic": "x", "correct": True}]}, "assessment.attempt.submit"),
    ],
)
async def test_a_body_naming_an_identity_is_422(app, path, body, permission, field):
    assert (await post(app, path, {**body, field: "someone"}, permission)).status_code == 422
    assert app.state.memory.calls == []


@pytest.mark.parametrize(
    "body",
    [
        {"observations": []},
        {"observations": [{"topic": "x", "correct": True}] * 201},
        {"observations": [{"topic": "x" * 256, "correct": True}]},
        {"observations": [{"topic": "", "correct": True}]},
        {"observations": [{"topic": "x", "correct": True, "user_id": "u"}]},
    ],
    ids=["empty", "too-many", "long-topic", "empty-topic", "identity-in-item"],
)
async def test_upsert_input_is_bounded(app, body):
    assert (await post(app, UPSERT, body, "assessment.attempt.submit")).status_code == 422


async def test_query_topics_are_bounded(app):
    assert (await post(app, QUERY, {"topics": ["t"] * 51}, "ai.use")).status_code == 422
