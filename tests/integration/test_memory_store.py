"""MemoryService against real PostgreSQL (M5 design §3 memory semantics).

Needs DATABASE_URL pointing at a database migrated with `alembic upgrade
head` (the upsert relies on the unique key from a7c2e5d10b41).
"""

import asyncio
import os
import uuid

import pytest

from app.core import tenant
from app.db.session import get_async_session_local, reset_engine
from app.services.memory.memory_service import MemoryService

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"), reason="needs PostgreSQL"
)


@pytest.fixture
async def scope():
    yield str(uuid.uuid4()), str(uuid.uuid4())
    await reset_engine()


# Each call binds the scope itself, as the signed route would for a request.
async def record(tenant_id, user_id, observations):
    tenant.set_request_scope(tenant_id, user_id)
    async with get_async_session_local()() as db:
        return await MemoryService(db).record(user_id, observations)


async def query(tenant_id, user_id, topics=None):
    tenant.set_request_scope(tenant_id, user_id)
    async with get_async_session_local()() as db:
        return await MemoryService(db).query(user_id, topics)


async def test_observations_move_mastery_and_count_attempts(scope):
    t, user = scope
    rows = await record(t, user, [("algebra", True), ("algebra", True), ("cells", False)])
    by_topic = {r.topic: r for r in rows}
    assert by_topic["algebra"].mastery_score == pytest.approx(0.7)
    assert by_topic["algebra"].attempts == 2
    assert by_topic["cells"].mastery_score == pytest.approx(0.45)


async def test_mastery_is_clamped(scope):
    t, user = scope
    await record(t, user, [("easy", True)] * 8)
    await record(t, user, [("hard", False)] * 15)
    rows = {r.topic: r.mastery_score for r in await query(t, user)}
    assert rows == {"easy": pytest.approx(1.0), "hard": pytest.approx(0.0)}


async def test_memory_is_isolated_by_user_and_tenant(scope):
    tenant_a, user = scope
    await record(tenant_a, user, [("algebra", True)])
    assert await query(tenant_a, str(uuid.uuid4())) == []
    assert await query(str(uuid.uuid4()), user) == []
    assert [r.topic for r in await query(tenant_a, user, ["algebra", "missing"])] == ["algebra"]


async def test_concurrent_upserts_keep_one_row_and_every_attempt(scope):
    t, user = scope
    await asyncio.gather(*(record(t, user, [("race", True)]) for _ in range(10)))
    rows = await query(t, user, ["race"])
    assert len(rows) == 1
    assert rows[0].attempts == 10
    assert rows[0].mastery_score == pytest.approx(1.0)
