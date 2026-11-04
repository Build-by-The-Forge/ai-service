"""asset_scopes and PgScopeStore against real PostgreSQL (M3-D).

Assignments apply only in version order, an asset another tenant holds is
refused, and the table refuses a scope or version outside its rules.
"""

import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.core.rag.scope import KnowledgeScope
from app.db.session import get_async_session_local, initialize_database, reset_engine
from app.services.rag.pg_index import PgChunkIndex
from app.services.rag.scopes import AssetHeldByAnotherTenantError, PgScopeStore

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"), reason="needs PostgreSQL"
)
INSTITUTIONAL, PERSONAL, UNKNOWN = (
    KnowledgeScope.INSTITUTIONAL,
    KnowledgeScope.PERSONAL,
    KnowledgeScope.UNKNOWN,
)


@pytest.fixture
async def store():
    await initialize_database()
    yield PgScopeStore(get_async_session_local())
    await reset_engine()


def ids() -> tuple[str, str]:
    return str(uuid.uuid4()), str(uuid.uuid4())


async def test_an_assignment_applies_only_above_the_stored_version(store):
    tenant, asset = ids()
    first = await store.assign(tenant_id=tenant, asset_id=asset, scope=PERSONAL, version=2)
    assert (first.scope, first.version) == (PERSONAL, 2)
    # A replay, and an older assignment arriving late: nothing changes, and
    # the stored state comes back so the sender can stop.
    for version in (2, 1):
        again = await store.assign(
            tenant_id=tenant, asset_id=asset, scope=INSTITUTIONAL, version=version
        )
        assert (again.scope, again.version) == (PERSONAL, 2)
    raised = await store.assign(tenant_id=tenant, asset_id=asset, scope=INSTITUTIONAL, version=3)
    assert (raised.scope, raised.version) == (INSTITUTIONAL, 3)
    lowered = await store.assign(tenant_id=tenant, asset_id=asset, scope=UNKNOWN, version=4)
    assert (lowered.scope, lowered.version) == (UNKNOWN, 4)


async def test_the_version_rule_holds_past_five_runs(store):
    # Repeated past five executions, where a cached generic plan could differ.
    tenant, asset = ids()
    for version in range(1, 9):
        state = await store.assign(
            tenant_id=tenant, asset_id=asset, scope=INSTITUTIONAL, version=version
        )
        assert state.version == version
        stale = await store.assign(tenant_id=tenant, asset_id=asset, scope=UNKNOWN, version=1)
        assert (stale.scope, stale.version) == (INSTITUTIONAL, version)


async def test_an_asset_fenced_by_another_tenant_is_refused(store):
    owner, asset = ids()
    other = str(uuid.uuid4())
    await PgChunkIndex(get_async_session_local()).raise_fence(
        tenant_id=owner, asset_id=asset, run_number=1, attempt=1
    )
    with pytest.raises(AssetHeldByAnotherTenantError):
        await store.assign(tenant_id=other, asset_id=asset, scope=INSTITUTIONAL, version=1)
    # The owner can assign it.
    state = await store.assign(tenant_id=owner, asset_id=asset, scope=INSTITUTIONAL, version=1)
    assert state.scope == INSTITUTIONAL
    async with get_async_session_local()() as db:
        rows = (
            await db.execute(
                text("SELECT tenant_id FROM asset_scopes WHERE asset_id = :a"), {"a": asset}
            )
        ).all()
    assert [r.tenant_id for r in rows] == [owner]


async def test_an_asset_scoped_by_another_tenant_is_refused(store):
    owner, asset = ids()
    await store.assign(tenant_id=owner, asset_id=asset, scope=PERSONAL, version=1)
    with pytest.raises(AssetHeldByAnotherTenantError):
        await store.assign(
            tenant_id=str(uuid.uuid4()), asset_id=asset, scope=INSTITUTIONAL, version=7
        )


@pytest.mark.parametrize(
    ("scope", "version"), [("public", 1), ("", 1), ("INSTITUTIONAL", 1), ("unknown", 0)]
)
async def test_the_table_refuses_a_scope_or_version_outside_its_rules(store, scope, version):
    tenant, asset = ids()
    with pytest.raises(IntegrityError):
        async with get_async_session_local()() as db, db.begin():
            await db.execute(
                text(
                    "INSERT INTO asset_scopes (tenant_id, asset_id, scope, version)"
                    " VALUES (:t, :a, :s, :v)"
                ),
                {"t": tenant, "a": asset, "s": scope, "v": version},
            )
