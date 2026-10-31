"""One update per event, and contributions the aggregate can be rebuilt from,
against real PostgreSQL. The memory gate stays closed in production; tests run
with it on, as development can."""

import asyncio
import os
import uuid

import pytest
from sqlalchemy import func, select

from app.core import tenant
from app.db.models.learning_memory import (
    LearningMemory,
    LearningMemoryContribution,
    LearningMemoryEvent,
)
from app.db.session import get_async_session_local, initialize_database, reset_engine
from app.services.memory.memory_service import (
    LearningMemoryEventConflict,
    LearningMemoryHistoryIncomplete,
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


SOURCE = ("assessment_attempt", "attempt-1")


async def test_an_unchanged_source_on_a_repeat_succeeds_and_changes_nothing(scope):
    t, user = scope
    await call(t, user, "record", [RIGHT], event_id="e1", source=SOURCE)
    before = await snapshot(t, user)
    await call(t, user, "record", [RIGHT], event_id="e1", source=SOURCE)
    assert await snapshot(t, user) == before
    assert await contributions(t, user) == 1


@pytest.mark.parametrize(
    "changed",
    [
        ("document", "attempt-1"),
        ("assessment_attempt", "attempt-2"),
        None,
    ],
    ids=["changed_kind", "changed_id", "removed"],
)
async def test_a_repeat_with_a_different_source_is_refused_and_changes_nothing(scope, changed):
    t, user = scope
    await call(t, user, "record", [RIGHT], event_id="e1", source=SOURCE)
    before = await snapshot(t, user)
    with pytest.raises(LearningMemoryEventConflict):
        await call(t, user, "record", [RIGHT], event_id="e1", source=changed)
    assert await snapshot(t, user) == before
    assert await contributions(t, user) == 1


async def test_a_repeat_cannot_add_a_source_the_event_did_not_have(scope):
    t, user = scope
    await call(t, user, "record", [RIGHT], event_id="e1")
    with pytest.raises(LearningMemoryEventConflict):
        await call(t, user, "record", [RIGHT], event_id="e1", source=SOURCE)


@pytest.mark.parametrize("change", ["delete", "revise"])
async def test_after_a_change_a_late_retry_still_needs_the_same_source(scope, change):
    # The same retry changes nothing and succeeds; another source is refused.
    t, user = scope
    await call(t, user, "record", [RIGHT], event_id="e1", source=SOURCE)
    if change == "delete":
        await call(t, user, "delete_event", "e1")
    else:
        await call(t, user, "revise_event", "e1", [WRONG])
    after = await snapshot(t, user)

    await call(t, user, "record", [RIGHT], event_id="e1", source=SOURCE)
    assert await snapshot(t, user) == after
    with pytest.raises(LearningMemoryEventConflict):
        await call(t, user, "record", [RIGHT], event_id="e1", source=("document", "d1"))
    assert await snapshot(t, user) == after


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


# --- the per-student lock, with the interleaving forced ---------------------------------


class PausesBeforeTheAggregate(MemoryService):
    """Holds a record after its event and contributions, before the aggregate."""

    def __init__(self, db, inserted: asyncio.Event, resume: asyncio.Event) -> None:
        super().__init__(db)
        self.inserted, self.resume = inserted, resume

    async def _add_contributions(self, *args, **kwargs):
        await super()._add_contributions(*args, **kwargs)
        self.inserted.set()
        await self.resume.wait()


class PausesWithoutTheLock(PausesBeforeTheAggregate):
    async def _lock_student(self, tenant_id, user_id):
        return None


class NoLock(MemoryService):
    async def _lock_student(self, tenant_id, user_id):
        return None


async def _first_wrong_then_right(t, user, slow_class, fast_class):
    """Mastery starts at the ceiling. The slow record (wrong) takes the earlier
    contribution number and pauses; the fast one (right) then runs. Clamping
    makes order matter: wrong then right ends at 1.0, right then wrong at 0.95."""
    for index in range(6):
        await call(t, user, "record", [RIGHT], event_id=f"seed{index}")
    assert (await snapshot(t, user))[0][1] == 1.0
    inserted, resume = asyncio.Event(), asyncio.Event()

    async def slow():
        tenant.set_request_scope(t, user)
        async with get_async_session_local()() as db:
            await slow_class(db, inserted, resume).record(user, [WRONG], event_id="slow")

    async def fast():
        await inserted.wait()
        tenant.set_request_scope(t, user)
        async with get_async_session_local()() as db:
            await fast_class(db).record(user, [RIGHT], event_id="fast")

    slow_task, fast_task = asyncio.create_task(slow()), asyncio.create_task(fast())
    await inserted.wait()
    done, _ = await asyncio.wait({fast_task}, timeout=1.0)
    finished_while_paused = bool(done)
    resume.set()
    await asyncio.gather(slow_task, fast_task)
    return finished_while_paused


async def _history(t, user):
    """The outcomes in contribution order: what a recompute would fold."""
    async with get_async_session_local()() as db:
        rows = await db.execute(
            select(LearningMemoryContribution.correct)
            .where(
                LearningMemoryContribution.tenant_id == t,
                LearningMemoryContribution.user_id == user,
            )
            .order_by(LearningMemoryContribution.event_row_id, LearningMemoryContribution.position)
        )
        return list(rows.scalars().all())


async def test_with_the_lock_memory_matches_its_ordered_history(scope):
    t, user = scope

    finished_while_paused = await _first_wrong_then_right(
        t, user, PausesBeforeTheAggregate, MemoryService
    )

    assert finished_while_paused is False  # the second record waited
    running = await snapshot(t, user)
    assert running == [("algebra", round(fold_mastery(await _history(t, user)), 6), 8)]
    assert running[0][1] == 1.0
    await call(t, user, "recompute", "algebra")
    assert await snapshot(t, user) == running


async def test_without_the_lock_memory_disagrees_with_its_ordered_history(scope):
    # The failure the lock rules out: the two applied in the opposite order to
    # their contribution numbers, so a recompute changes the student's mastery.
    t, user = scope

    finished_while_paused = await _first_wrong_then_right(t, user, PausesWithoutTheLock, NoLock)

    assert finished_while_paused is True
    running = await snapshot(t, user)
    assert running[0][1] == 0.95
    assert fold_mastery(await _history(t, user)) == pytest.approx(1.0)
    await call(t, user, "recompute", "algebra")
    assert (await snapshot(t, user))[0][1] == 1.0


# --- memory from before contributions existed -------------------------------------------


async def _legacy_row(t, user, topic="algebra"):
    tenant.set_request_scope(t, user)
    async with get_async_session_local()() as db:
        db.add(
            LearningMemory(
                tenant_id=t,
                user_id=user,
                topic=topic,
                mastery_score=0.8,
                attempts=7,
                last_interaction="2026-01-01T00:00:00+00:00",
                history_complete=False,
            )
        )
        await db.commit()


async def test_recomputing_a_legacy_row_is_refused_and_keeps_it(scope):
    t, user = scope
    await _legacy_row(t, user)
    with pytest.raises(LearningMemoryHistoryIncomplete):
        await call(t, user, "recompute", "algebra")
    assert await snapshot(t, user) == [("algebra", 0.8, 7)]


async def test_a_new_event_on_a_legacy_row_applies_and_it_stays_incomplete(scope):
    # Its history is still missing, however much is added to it.
    t, user = scope
    await _legacy_row(t, user)
    await call(t, user, "record", [RIGHT], event_id="e1")
    assert await snapshot(t, user) == [("algebra", 0.9, 8)]
    assert await contributions(t, user) == 1
    with pytest.raises(LearningMemoryHistoryIncomplete):
        await call(t, user, "recompute", "algebra")
    assert await snapshot(t, user) == [("algebra", 0.9, 8)]


@pytest.mark.parametrize(
    "method,args",
    [("delete_event", ("e1",)), ("revise_event", ("e1", [WRONG]))],
    ids=["delete", "revise"],
)
async def test_changing_an_event_on_a_legacy_row_is_refused_and_changes_nothing(
    scope, method, args
):
    t, user = scope
    await _legacy_row(t, user)
    await call(t, user, "record", [RIGHT], event_id="e1")

    with pytest.raises(LearningMemoryHistoryIncomplete):
        await call(t, user, method, *args)

    assert await snapshot(t, user) == [("algebra", 0.9, 8)]
    assert await contributions(t, user) == 1
    # It wasn't recorded as changed either, so a repeat is still just a repeat.
    await call(t, user, "record", [RIGHT], event_id="e1")
    assert await snapshot(t, user) == [("algebra", 0.9, 8)]


async def test_a_legacy_row_does_not_stop_other_topics_recomputing(scope):
    t, user = scope
    await _legacy_row(t, user)
    await call(t, user, "record", [("cells", True)], event_id="e1")
    await call(t, user, "recompute", "cells")
    assert (await snapshot(t, user, "cells"))[0][2] == 1


async def test_rows_written_now_are_marked_complete(scope):
    t, user = scope
    [row] = await call(t, user, "record", [RIGHT], event_id="e1")
    assert row.history_complete is True
