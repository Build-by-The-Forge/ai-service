"""The in-memory retrieval engine, for tests only: a deterministic hash
embedder, an in-memory vector store and an in-memory BM25 index. The app's
default engine is the durable one (retrieval.durable_retrieval_engine).

Both stores read each document's scope from ``scopes`` (M3-D), as the
durable index reads asset_scopes: a document never assigned one is
``unknown`` and never returned. ``institutional`` assigns a document the
scope tenant-wide retrieval returns, as the backend's signed assignment
would."""

from app.core.rag.scope import KnowledgeScope
from app.services.rag.embedder import MockEmbeddingAdapter
from app.services.rag.lexical import BM25Index
from app.services.rag.retrieval import HybridRetrievalEngine
from app.services.rag.scopes import InMemoryScopes
from app.services.rag.vector_store import InMemoryVectorStore


def in_memory_engine(scopes: InMemoryScopes | None = None) -> HybridRetrievalEngine:
    scopes = scopes if scopes is not None else InMemoryScopes()
    return HybridRetrievalEngine(
        embedder=MockEmbeddingAdapter(),
        vector_store=InMemoryVectorStore(scopes=scopes),
        lexical_index=BM25Index(scopes=scopes),
    )


async def institutional(scopes: InMemoryScopes, tenant_id: str, *asset_ids: str) -> None:
    for asset_id in asset_ids:
        await scopes.assign(
            tenant_id=tenant_id, asset_id=asset_id, scope=KnowledgeScope.INSTITUTIONAL, version=1
        )


async def stored_scope(
    tenant_id: str, asset_id: str, scope: str = "institutional", version: int = 1
) -> None:
    """Write a document's scope row directly, as an accepted assignment
    leaves it. For tests of the durable index, whose fixed asset ids would
    otherwise meet the store's check that no other tenant holds the asset;
    PgScopeStore's own rules are tested in test_asset_scopes.py."""
    from sqlalchemy import text

    from app.db.session import get_async_session_local

    async with get_async_session_local()() as db, db.begin():
        await db.execute(
            text(
                "INSERT INTO asset_scopes (tenant_id, asset_id, scope, version)"
                " VALUES (:t, :a, :s, :v) ON CONFLICT (tenant_id, asset_id)"
                " DO UPDATE SET scope = EXCLUDED.scope, version = EXCLUDED.version"
            ),
            {"t": tenant_id, "a": asset_id, "s": scope, "v": version},
        )
