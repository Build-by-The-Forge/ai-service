"""Who owns an asset, on the dispatch route against real PostgreSQL (M3-D PR 7).

An asset another tenant already holds, by its ingestion fence or by its scope,
is refused with 403 before anything is queued, at version zero as at any other
version: nothing is queued, no fence moves, no scope row is written, and the
claim on the idempotency key is released. Two tenants naming one new asset at
once take turns: exactly one gets it.

The matrix: (scope_version zero or not) x (scope row: own, none, another
tenant's) x (fence: own, none, another tenant's) x (first dispatch, duplicate,
retry, concurrent from two tenants). The dangerous cells are tested; the
ordinary ones are the controls.
"""

import asyncio
import os
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from app.core.idempotency import InMemoryIdempotencyStore, set_idempotency_store
from app.core.rag.scope import KnowledgeScope
from app.core.replay_guard import RedisReplayGuard
from app.db.models.asset_scope import AssetScope
from app.db.models.ingestion_fence import IngestionFence
from app.db.session import get_async_session_local, initialize_database, reset_engine
from app.main import create_app
from app.services.rag.index_fence import set_index_fence
from app.services.rag.pg_index import PgChunkIndex
from app.services.rag.scopes import PgScopeStore, set_scope_store
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
pytestmark = [
    pytest.mark.skipif(
        not os.getenv("DATABASE_URL", "").startswith("postgresql"), reason="needs PostgreSQL"
    ),
    pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL not set"),
]


# Cases that are wrong on the code before the ownership fix are expected to
# fail there (strictly: a pass is an error). The fix commit makes this the
# identity, so they run as ordinary tests.
def refused_by_the_fix(test):
    return pytest.mark.xfail(strict=True, reason="fails before the ownership fix")(test)


ZERO = pytest.param(
    0, "unknown", marks=pytest.mark.xfail(strict=True, reason="fails before the ownership fix")
)


class RecordingDispatcher:
    def __init__(self) -> None:
        self.tasks: list[dict] = []

    async def dispatch_ingest(self, **kwargs):
        self.tasks.append(kwargs)


@pytest.fixture
async def world():
    guard = RedisReplayGuard.connect(REDIS_URL, timeout=2.0)
    await initialize_database()
    set_idempotency_store(InMemoryIdempotencyStore())
    set_scope_store(PgScopeStore())
    set_index_fence(PgChunkIndex())
    app = create_app(replay_guard=guard)
    recorder = RecordingDispatcher()
    app.dependency_overrides[get_task_dispatcher] = lambda: recorder
    yield app, recorder
    await guard.close()
    await reset_engine()


async def dispatch(
    app,
    tenant: str,
    asset: str,
    *,
    version: int = 0,
    scope: str = "unknown",
    key: str | None = None,
    job: str | None = None,
):
    job = job or f"job-{uuid.uuid4().hex[:12]}"
    body = dispatch_body(
        asset_id=asset,
        job_id=job,
        idempotency_key=key or f"{job}-r1-a1",
        scope=scope,
        scope_version=version,
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://ai-service") as http:
        return await http.post(
            DISPATCH_PATH, content=body, headers=signed_headers(body, context=user_context(tenant))
        )


def new_asset() -> str:
    return str(uuid.uuid4())


async def fences(asset: str) -> dict[str, tuple[int, int]]:
    async with get_async_session_local()() as db:
        rows = await db.execute(
            select(
                IngestionFence.tenant_id, IngestionFence.run_number, IngestionFence.attempt
            ).where(IngestionFence.asset_id == asset)
        )
        return {str(r.tenant_id): (r.run_number, r.attempt) for r in rows}


async def scopes(asset: str) -> dict[str, tuple[str, int]]:
    async with get_async_session_local()() as db:
        rows = await db.execute(
            select(AssetScope.tenant_id, AssetScope.scope, AssetScope.version).where(
                AssetScope.asset_id == asset
            )
        )
        return {str(r.tenant_id): (r.scope, r.version) for r in rows}


async def give_a_fence(tenant: str, asset: str, run: int = 1, attempt: int = 1) -> None:
    await PgChunkIndex().raise_fence(
        tenant_id=tenant, asset_id=asset, run_number=run, attempt=attempt
    )


async def give_a_scope(
    tenant: str, asset: str, scope=KnowledgeScope.PERSONAL, version: int = 1
) -> None:
    await PgScopeStore().assign(tenant_id=tenant, asset_id=asset, scope=scope, version=version)


# --- another tenant owns the asset: refused, whatever the version --------------------


@pytest.mark.parametrize("version, scope", [ZERO, (1, "personal"), (7, "institutional")])
@pytest.mark.parametrize("held_by", ["fence", "scope", "both"])
async def test_an_asset_another_tenant_holds_is_refused_and_nothing_changes(
    world, version, scope, held_by
):
    app, recorder = world
    asset = new_asset()
    if held_by in ("fence", "both"):
        await give_a_fence(TENANT_B, asset, run=3, attempt=2)
    if held_by in ("scope", "both"):
        await give_a_scope(TENANT_B, asset, KnowledgeScope.PERSONAL, 4)
    before_fences, before_scopes = await fences(asset), await scopes(asset)

    response = await dispatch(app, TENANT_A, asset, version=version, scope=scope)

    assert response.status_code == 403, response.text
    assert recorder.tasks == []
    assert await fences(asset) == before_fences  # no fence of A's, and B's didn't move
    assert await scopes(asset) == before_scopes  # no scope row of A's, and B's didn't move
    assert TENANT_A not in await fences(asset) and TENANT_A not in await scopes(asset)


# --- controls: the tenant's own asset, and an unowned one ------------------------------


@pytest.mark.parametrize("version, scope", [(0, "unknown"), (2, "personal")])
async def test_the_tenants_own_asset_is_accepted_at_any_version(world, version, scope):
    app, recorder = world
    asset = new_asset()
    await give_a_fence(TENANT_A, asset)
    await give_a_scope(TENANT_A, asset, KnowledgeScope.UNKNOWN, 1)

    response = await dispatch(app, TENANT_A, asset, version=version, scope=scope)

    assert response.status_code == 202, response.text
    assert len(recorder.tasks) == 1
    assert TENANT_B not in await fences(asset) and TENANT_B not in await scopes(asset)


@refused_by_the_fix
async def test_a_version_zero_dispatch_for_a_new_asset_takes_the_fence_and_writes_no_scope(world):
    app, recorder = world
    asset = new_asset()

    response = await dispatch(app, TENANT_A, asset)

    assert response.status_code == 202, response.text
    assert len(recorder.tasks) == 1
    assert list(await fences(asset)) == [TENANT_A]
    assert await scopes(asset) == {}
    # And now the asset is A's: B is refused at any version.
    for version, scope in ((0, "unknown"), (3, "institutional")):
        assert (
            await dispatch(app, TENANT_B, asset, version=version, scope=scope)
        ).status_code == 403
    assert len(recorder.tasks) == 1 and list(await fences(asset)) == [TENANT_A]


# --- duplicates and retries cannot get around the rule --------------------------------


@refused_by_the_fix
async def test_another_tenant_repeating_the_owners_key_gets_a_refusal_never_the_owners_task(world):
    app, recorder = world
    asset = new_asset()
    first = await dispatch(app, TENANT_A, asset, key="shared-key-1", job="job-shared-1")
    assert first.status_code == 202

    # B uses A's idempotency key and A's asset: its own key namespace has no
    # task, and the asset isn't its own.
    second = await dispatch(app, TENANT_B, asset, key="shared-key-1", job="job-shared-1")

    assert second.status_code == 403, second.text
    assert first.json()["external_task_id"] not in second.text
    assert len(recorder.tasks) == 1 and list(await fences(asset)) == [TENANT_A]


async def test_the_owners_own_duplicate_is_still_a_duplicate(world):
    app, recorder = world
    asset = new_asset()
    first = await dispatch(app, TENANT_A, asset, key="own-key-1", job="job-own-1")
    again = await dispatch(app, TENANT_A, asset, key="own-key-1", job="job-own-1")
    assert first.status_code == again.status_code == 202
    assert again.json()["status"] == "DUPLICATE"
    assert again.json()["external_task_id"] == first.json()["external_task_id"]
    assert len(recorder.tasks) == 1


@refused_by_the_fix
async def test_a_refused_dispatch_leaves_its_key_free_for_a_retry_once_the_asset_is_its_own(world):
    """A refusal releases nothing it didn't claim and claims nothing: the same
    key is usable later for an asset the tenant does own."""
    app, recorder = world
    other_asset, own_asset = new_asset(), new_asset()
    await give_a_fence(TENANT_B, other_asset)

    refused = await dispatch(app, TENANT_A, other_asset, key="reused-key")
    assert refused.status_code == 403

    accepted = await dispatch(app, TENANT_A, own_asset, key="reused-key")
    assert accepted.status_code == 202, accepted.text
    assert len(recorder.tasks) == 1


# --- two tenants naming one new asset at once -----------------------------------------


@pytest.mark.parametrize(
    "version",
    [
        pytest.param(
            0, marks=pytest.mark.xfail(strict=True, reason="fails before the ownership fix")
        ),
        pytest.param(
            1, marks=pytest.mark.xfail(strict=False, reason="a race the old code usually loses")
        ),
    ],
)
async def test_two_tenants_dispatching_one_new_asset_at_once_get_exactly_one_winner(world, version):
    """The fence is claimed under the asset's lock, so whichever tenant's
    transaction takes it first owns it, and the other is refused: never two
    owners, never two tasks. Repeated, because the order is the scheduler's."""
    app, recorder = world
    scope = "unknown" if version == 0 else "personal"
    for _ in range(5):
        asset = new_asset()
        a, b = await asyncio.gather(
            dispatch(app, TENANT_A, asset, version=version, scope=scope),
            dispatch(app, TENANT_B, asset, version=version, scope=scope),
        )
        assert sorted((a.status_code, b.status_code)) == [202, 403], (a.text, b.text)
        owners = set(await fences(asset)) | set(await scopes(asset))
        assert len(owners) == 1, owners
        winner = TENANT_A if a.status_code == 202 else TENANT_B
        assert owners == {winner}
    assert len(recorder.tasks) == 5


# --- the stores themselves ------------------------------------------------------------


@refused_by_the_fix
async def test_the_fence_refuses_an_asset_another_tenant_holds_and_writes_nothing(world):
    from app.services.rag.scopes import AssetHeldByAnotherTenantError

    asset = new_asset()
    await give_a_scope(TENANT_B, asset)
    with pytest.raises(AssetHeldByAnotherTenantError):
        await give_a_fence(TENANT_A, asset)
    assert await fences(asset) == {}


@refused_by_the_fix
async def test_current_refuses_an_asset_another_tenant_holds(world):
    from app.services.rag.scopes import AssetHeldByAnotherTenantError

    asset = new_asset()
    await give_a_fence(TENANT_B, asset)
    with pytest.raises(AssetHeldByAnotherTenantError):
        await PgScopeStore().current(tenant_id=TENANT_A, asset_id=asset)
    # The owner reads its own, and an unowned asset reads as unknown at version 0.
    assert (await PgScopeStore().current(tenant_id=TENANT_B, asset_id=asset)).version == 0
    unowned = await PgScopeStore().current(tenant_id=TENANT_A, asset_id=new_asset())
    assert (unowned.scope, unowned.version) == (KnowledgeScope.UNKNOWN, 0)
