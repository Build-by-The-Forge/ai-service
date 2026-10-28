"""The ingestion_fences backfill (P3-6 §11) against real PostgreSQL: each
indexed asset gets a fence at its chunks' (run, attempt), and an existing
fence is kept."""

import importlib.util
import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text

from app.db.session import get_async_session_local, initialize_database, reset_engine

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"), reason="needs PostgreSQL"
)
MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "f1a3c6e8b2d5_backfill_ingestion_fences.py"
)
VECTOR = "[" + ",".join(["1"] + ["0"] * 383) + "]"


def backfill_sql() -> str:
    spec = importlib.util.spec_from_file_location("backfill", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BACKFILL


async def test_each_indexed_asset_is_fenced_at_its_chunks_attempt():
    await initialize_database()
    tenant = str(uuid.uuid4())
    chunks = [  # asset, run, attempt
        ("a1", 1, 1),
        ("a1", 1, 1),
        ("a2", 1, 2),
        ("a2", 2, 1),  # a later run wins over a higher attempt of an earlier one
        ("a3", 3, 1),  # already fenced higher by a dispatch: kept
    ]
    try:
        async with get_async_session_local()() as db, db.begin():
            for i, (asset, run, attempt) in enumerate(chunks):
                await db.execute(
                    text(
                        "INSERT INTO document_chunks (chunk_id, tenant_id, asset_id, run_number,"
                        " attempt, chunk_index, page_start, page_end, content, embedding,"
                        " embedding_model) VALUES (:id, :t, :a, :r, :n, :i, 1, 1, 'x',"
                        " CAST(:v AS vector), 'm')"
                    ),
                    {
                        "id": str(uuid.uuid4()),
                        "t": tenant,
                        "a": asset,
                        "r": run,
                        "n": attempt,
                        "i": i,
                        "v": VECTOR,
                    },
                )
            await db.execute(
                text(
                    "INSERT INTO ingestion_fences (tenant_id, asset_id, run_number, attempt)"
                    " VALUES (:t, 'a3', 3, 4)"
                ),
                {"t": tenant},
            )
            await db.execute(text(backfill_sql()))
            await db.execute(text(backfill_sql()))  # idempotent
            fences = (
                await db.execute(
                    text(
                        "SELECT asset_id, run_number, attempt FROM ingestion_fences"
                        " WHERE tenant_id = :t ORDER BY asset_id"
                    ),
                    {"t": tenant},
                )
            ).all()
        assert [tuple(f) for f in fences] == [("a1", 1, 1), ("a2", 2, 1), ("a3", 3, 4)]
    finally:
        async with get_async_session_local()() as db, db.begin():
            await db.execute(
                text("DELETE FROM document_chunks WHERE tenant_id = :t"), {"t": tenant}
            )
            await db.execute(
                text("DELETE FROM ingestion_fences WHERE tenant_id = :t"), {"t": tenant}
            )
        await reset_engine()
