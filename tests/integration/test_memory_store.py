"""MemoryService against real PostgreSQL (M5 design §3 memory semantics).

Needs DATABASE_URL pointing at PostgreSQL. The fixture creates the AI-owned
tables if they are missing; the model carries the unique key that migration
a7c2e5d10b41 adds, which the upsert relies on.
"""

import asyncio
import os
import uuid

import pytest

from app.core import tenant
from app.core.config import settings
from app.core.learning_memory_gate import LearningMemoryDisabled
from app.db.session import get_async_session_local, initialize_database, reset_engine
from app.services.ai.orchestrator import AIOrchestrator
from app.services.memory.memory_service import MemoryService
from app.services.rag.evidence import Evidence, Passage

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"), reason="needs PostgreSQL"
)


@pytest.fixture
async def scope():
    # The AI-owned tables, including the unique key the upsert needs, whether
    # or not this database was migrated (CI's ci.yml job does not run Alembic).
    await initialize_database()
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


class RecordingLLM:
    def __init__(self):
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return "Water moves [1]."


class OnePassage:
    async def evidence(self, query):
        passage = Passage(
            number=1, text="Osmosis.", chunk_id="c1", document_id="d1", page_start=1, page_end=1
        )
        return Evidence(query=query, passages=(passage,))


async def test_stored_memories_are_unreachable_while_the_gate_is_closed(scope, monkeypatch):
    t, user = scope
    marker = f"stored-{uuid.uuid4().hex[:8]}"
    await record(t, user, [(marker, True)])

    monkeypatch.setattr(settings, "LEARNING_MEMORY_ENABLED", False)
    tenant.set_request_scope(t, user)
    async with get_async_session_local()() as db:
        service = MemoryService(db)
        with pytest.raises(LearningMemoryDisabled):
            await service.query(user, [marker])
        with pytest.raises(LearningMemoryDisabled):
            await service.get_user_memory(user)
        with pytest.raises(LearningMemoryDisabled):
            await service.record(user, [(marker, True)])
        with pytest.raises(LearningMemoryDisabled):
            await service.update_mastery(user, marker, 0.1)

        llm = RecordingLLM()
        chat = await AIOrchestrator(llm=llm, retriever=OnePassage(), memory=service).chat(user, "q")
    assert chat["memory_used"] == ""
    assert chat["grounded"] is True
    assert marker not in llm.prompts[0]

    # The rows are kept, unchanged: the refused write above changed nothing.
    monkeypatch.setattr(settings, "LEARNING_MEMORY_ENABLED", True)
    [row] = await query(t, user, [marker])
    assert row.attempts == 1
    assert row.mastery_score == pytest.approx(0.6)
