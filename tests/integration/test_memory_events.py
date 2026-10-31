"""One update per event, and contributions the aggregate can be rebuilt from,
against real PostgreSQL. The memory gate stays closed in production; tests run
with it on, as development can."""

import asyncio
import os
import uuid

import pytest
from sqlalchemy import func, select

from app.core import tenant
from app.db.models.learning_memory import LearningMemoryContribution, LearningMemoryEvent
from app.db.session import get_async_session_local, initialize_database, reset_engine
from app.services.memory.memory_service import (
    LearningMemoryEventConflict,
    MemoryService,
    fold_mastery,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"), reason="needs PostgreSQL"
)

RIGHT, WRONG = ("algebra", True), ("algebra", False)


@pytest.fixture
async def scope():
    await initialize_database()
    yield str(uuid.uuid4()), str(uuid.uuid4())
    await reset_engine()


async def call(t, user, method, *args, **kwargs):
    tenant.set_request_scope(t, user)
    async with get_async_session_local()() as db:
        return await getattr(MemoryService(db), method)(user, *args, **kwargs)


async def snapshot(t, user, topic="algebra"):
    rows = await call(t, user, "query", [topic])
    return [(r.topic, round(r.mastery_score, 6), r.attempts) for r in rows]


async def contributions(t, user):
    async with get_async_session_local()() as db:
        statement = select(func.count()).where(
            LearningMemoryContribution.tenant_id == t, LearningMemoryContribution.user_id == user
        )
        return (await db.execute(statement)).scalar_one()


# --- one update per event ------------------------------------------------------------


async def test_a_repeated_event_succeeds_and_changes_nothing(scope):
    t, user = scope
    first = await call(t, user, "record", [RIGHT], event_id="e1")
    before = await snapshot(t, user)

    repeat = await call(t, user, "record", [RIGHT], event_id="e1")

    assert [r.attempts for r in repeat] == [r.attempts for r in first] == [1]
    assert await snapshot(t, user) == before
    assert await contributions(t, user) == 1


async def test_many_repeats_on_one_connection_keep_working(scope):
    # Past PostgreSQL's move to a generic plan, where a statement that depends
    # on plan-time matching would start failing.
    t, user = scope
    tenant.set_request_scope(t, user)
    async with get_async_session_local()() as db:
        service = MemoryService(db)
        for _ in range(12):
            await service.record(user, [RIGHT], event_id="e1")
    assert (await snapshot(t, user))[0][2] == 1


async def test_concurrent_repeats_apply_once(scope):
    t, user = scope
    await asyncio.gather(*(call(t, user, "record", [RIGHT], event_id="e1") for _ in range(8)))
    assert (await snapshot(t, user))[0][2] == 1
    assert await contributions(t, user) == 1


async def test_different_events_all_count(scope):
    t, user = scope
    await asyncio.gather(*(call(t, user, "record", [RIGHT], event_id=f"e{i}") for i in range(8)))
    assert (await snapshot(t, user))[0][2] == 8


async def test_without_an_event_id_a_repeat_counts_twice(scope):
    # What callers get until L4 sends IDs, stated rather than hidden.
    t, user = scope
    await call(t, user, "record", [RIGHT])
    await call(t, user, "record", [RIGHT])
    assert (await snapshot(t, user))[0][2] == 2


async def test_an_event_id_with_a_different_body_is_refused_and_changes_nothing(scope):
    t, user = scope
    await call(t, user, "record", [RIGHT], event_id="e1")
    before = await snapshot(t, user)
    with pytest.raises(LearningMemoryEventConflict):
        await call(t, user, "record", [WRONG], event_id="e1")
    assert await snapshot(t, user) == before


async def test_another_students_event_id_is_refused_and_writes_nothing(scope):
    t, user = scope
    other = str(uuid.uuid4())
    await call(t, user, "record", [RIGHT], event_id="e1")
    with pytest.raises(LearningMemoryEventConflict):
        await call(t, other, "record", [RIGHT], event_id="e1")
    assert await snapshot(t, other) == []


async def test_the_same_event_id_in_another_tenant_is_its_own_event(scope):
    t, user = scope
    await call(t, user, "record", [RIGHT], event_id="e1")
    await call(str(uuid.uuid4()), user, "record", [RIGHT], event_id="e1")
    assert (await snapshot(t, user))[0][2] == 1


async def test_the_event_and_its_effect_commit_together(scope, monkeypatch):
    # A failure partway leaves no event, so the retry still applies.
    t, user = scope
    real = MemoryService._add_contributions

    async def failing(*args, **kwargs):
        raise RuntimeError("crash")

    monkeypatch.setattr(MemoryService, "_add_contributions", failing)
    with pytest.raises(RuntimeError):
        await call(t, user, "record", [RIGHT], event_id="e1")
    assert await snapshot(t, user) == []

    monkeypatch.setattr(MemoryService, "_add_contributions", real)
    await call(t, user, "record", [RIGHT], event_id="e1")
    assert (await snapshot(t, user))[0][2] == 1


async def test_the_event_records_its_generic_source_and_no_content(scope):
    t, user = scope
    await call(t, user, "record", [RIGHT], event_id="e1", source=("assessment_attempt", "a1"))
    async with get_async_session_local()() as db:
        event = (
            await db.execute(select(LearningMemoryEvent).where(LearningMemoryEvent.tenant_id == t))
        ).scalar_one()
    assert (event.source_kind, event.source_id) == ("assessment_attempt", "a1")
    assert not hasattr(event, "answer")


# --- contributions --------------------------------------------------------------------


async def test_deleting_one_contribution_leaves_the_others(scope):
    t, user = scope
    await call(t, user, "record", [RIGHT], event_id="e1")
    await call(t, user, "record", [WRONG], event_id="e2")
    await call(t, user, "record", [RIGHT], event_id="e3")

    await call(t, user, "delete_event", "e2")

    assert await snapshot(t, user) == [("algebra", round(fold_mastery([True, True]), 6), 2)]
    assert await contributions(t, user) == 2


async def test_deleting_the_only_contribution_removes_the_aggregate(scope):
    t, user = scope
    await call(t, user, "record", [RIGHT], event_id="e1")
    await call(t, user, "delete_event", "e1")
    assert await snapshot(t, user) == []


async def test_a_late_retry_cannot_bring_a_deleted_contribution_back(scope):
    t, user = scope
    await call(t, user, "record", [RIGHT], event_id="e1")
    await call(t, user, "delete_event", "e1")

    await call(t, user, "record", [RIGHT], event_id="e1")

    assert await snapshot(t, user) == []
    assert await contributions(t, user) == 0


async def test_recomputing_matches_the_running_aggregate_exactly(scope):
    t, user = scope
    # Enough rights to hit the clamp, where order changes the answer.
    outcomes = [True] * 8 + [False] * 3 + [True] * 2 + [False] * 15
    for index, correct in enumerate(outcomes):
        await call(t, user, "record", [("algebra", correct)], event_id=f"e{index}")
    running = await snapshot(t, user)

    await call(t, user, "recompute", "algebra")

    assert await snapshot(t, user) == running
    assert running == [("algebra", round(fold_mastery(outcomes), 6), len(outcomes))]


async def test_recomputing_twice_gives_the_same_answer(scope):
    t, user = scope
    await call(t, user, "record", [RIGHT, WRONG, RIGHT], event_id="e1")
    await call(t, user, "recompute", "algebra")
    first = await snapshot(t, user)
    await call(t, user, "recompute", "algebra")
    assert await snapshot(t, user) == first


async def test_concurrent_events_still_recompute_to_the_running_value(scope):
    # Clamping depends on order, so this holds only because the per-student
    # lock makes contribution order the order they applied.
    t, user = scope
    outcomes = [True] * 10 + [False] * 10
    await asyncio.gather(
        *(
            call(t, user, "record", [("algebra", c)], event_id=f"e{i}")
            for i, c in enumerate(outcomes)
        )
    )
    running = await snapshot(t, user)
    await call(t, user, "recompute", "algebra")
    assert await snapshot(t, user) == running


async def test_revising_replaces_the_contribution_with_no_duplicate(scope):
    t, user = scope
    await call(t, user, "record", [RIGHT], event_id="e1")
    await call(t, user, "record", [RIGHT], event_id="e2")

    await call(t, user, "revise_event", "e1", [WRONG])

    assert await snapshot(t, user) == [("algebra", round(fold_mastery([False, True]), 6), 2)]
    assert await contributions(t, user) == 2


async def test_revising_can_move_an_event_to_another_topic(scope):
    t, user = scope
    await call(t, user, "record", [RIGHT], event_id="e1")
    await call(t, user, "revise_event", "e1", [("cells", True)])
    assert await snapshot(t, user) == []
    assert (await snapshot(t, user, "cells"))[0][2] == 1


async def test_a_late_retry_cannot_undo_a_revision(scope):
    t, user = scope
    await call(t, user, "record", [RIGHT], event_id="e1")
    await call(t, user, "revise_event", "e1", [WRONG])
    revised = await snapshot(t, user)

    await call(t, user, "record", [RIGHT], event_id="e1")

    assert await snapshot(t, user) == revised


async def test_changing_another_students_event_is_refused(scope):
    t, user = scope
    other = str(uuid.uuid4())
    await call(t, user, "record", [RIGHT], event_id="e1")
    with pytest.raises(LookupError):
        await call(t, other, "delete_event", "e1")
    with pytest.raises(LookupError):
        await call(t, other, "revise_event", "e1", [WRONG])
    assert (await snapshot(t, user))[0][2] == 1
