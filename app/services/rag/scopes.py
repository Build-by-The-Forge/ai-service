"""Scope assignments (M3-D): the backend's decision on what each document
may be retrieved as, applied in version order.

- An assignment is (tenant, asset, scope, version). The version is the
  backend's and only rises per document. A row changes only to a higher
  version, so a replay, or an older assignment arriving late, changes
  nothing, and the stored state is still returned so the sender stops.
- The tenant is the verified call's. An asset another tenant holds, by its
  fence (every indexed asset has one) or by its scope, is refused.
- A document without a row is ``unknown``. Retrieval reads the row through
  a join (pg_index), never a copy.

The durable store is ``asset_scopes``; ``InMemoryScopes`` is for tests and
follows the same rules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rag.scope import RETRIEVABLE_SCOPES, KnowledgeScope
from app.db.models.asset_scope import AssetScope

_HELD_ELSEWHERE = text(
    "SELECT 1 FROM ingestion_fences WHERE asset_id = :asset_id AND tenant_id <> :tenant_id"
    " UNION ALL"
    " SELECT 1 FROM asset_scopes WHERE asset_id = :asset_id AND tenant_id <> :tenant_id"
    " LIMIT 1"
)


class ScopeStoreUnavailableError(Exception):
    """The scope store could not be read or written."""


class AssetHeldByAnotherTenantError(Exception):
    """The assignment names an asset that belongs to another tenant."""


@dataclass(frozen=True)
class ScopeState:
    asset_id: str
    scope: KnowledgeScope
    # 0 when the backend has never assigned the document a scope.
    version: int


class PgScopeStore:
    def __init__(self, session_factory: Callable[[], AsyncSession] | None = None) -> None:
        self._session_factory = session_factory

    def _session(self) -> AsyncSession:
        if self._session_factory is None:
            from app.db.session import get_async_session_local

            self._session_factory = get_async_session_local()
        return self._session_factory()

    async def assign(
        self, *, tenant_id: str, asset_id: str, scope: KnowledgeScope, version: int
    ) -> ScopeState:
        """Apply the assignment if its version is higher than the stored
        one; return the state now stored either way."""
        insert = pg_insert(AssetScope).values(
            tenant_id=tenant_id, asset_id=asset_id, scope=scope.value, version=version
        )
        stmt = insert.on_conflict_do_update(
            index_elements=[AssetScope.tenant_id, AssetScope.asset_id],
            set_={
                "scope": insert.excluded.scope,
                "version": insert.excluded.version,
                "updated_at": func.now(),
            },
            where=insert.excluded.version > AssetScope.version,
        )
        params = {"tenant_id": tenant_id, "asset_id": asset_id}
        try:
            async with self._session() as db, db.begin():
                if (await db.execute(_HELD_ELSEWHERE, params)).first() is not None:
                    raise AssetHeldByAnotherTenantError(asset_id)
                await db.execute(stmt)
                row = (
                    await db.execute(
                        select(AssetScope.scope, AssetScope.version).where(
                            AssetScope.tenant_id == tenant_id, AssetScope.asset_id == asset_id
                        )
                    )
                ).one()
        except SQLAlchemyError as exc:
            raise ScopeStoreUnavailableError(type(exc).__name__) from exc
        return ScopeState(asset_id, KnowledgeScope(row.scope), int(row.version))


class InMemoryScopes:
    """For tests only: the same rules as PgScopeStore, without a fence check."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], tuple[KnowledgeScope, int]] = {}

    async def assign(
        self, *, tenant_id: str, asset_id: str, scope: KnowledgeScope, version: int
    ) -> ScopeState:
        if any(a == asset_id and t != tenant_id for t, a in self.rows):
            raise AssetHeldByAnotherTenantError(asset_id)
        current = self.rows.get((tenant_id, asset_id))
        if current is None or version > current[1]:
            self.rows[(tenant_id, asset_id)] = (scope, version)
        stored, stored_version = self.rows[(tenant_id, asset_id)]
        return ScopeState(asset_id, stored, stored_version)

    def retrievable(self, tenant_id: str, asset_id: str) -> bool:
        row = self.rows.get((tenant_id, asset_id))
        return row is not None and row[0].value in RETRIEVABLE_SCOPES


_default_store: PgScopeStore | InMemoryScopes | None = None


def get_scope_store() -> PgScopeStore | InMemoryScopes:
    global _default_store
    if _default_store is None:
        _default_store = PgScopeStore()
    return _default_store


def set_scope_store(store: PgScopeStore | InMemoryScopes | None) -> None:
    global _default_store
    _default_store = store
