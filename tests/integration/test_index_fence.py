"""The index fence against real PostgreSQL (P3-6 §7.5, §14).

Only the attempt the backend most recently dispatched can write an asset's
index: ingestion_fences holds that (run, attempt), raised at dispatch and
never lowered, and replace_asset writes only if its attempt equals it,
under the fence row's lock. A superseded or late attempt changes nothing,
whatever it would write and whether or not the index has newer rows.
"""

import asyncio
import os
import uuid

import pytest
from sqlalchemy import select

from app.core.rag.errors import StaleIndexWriteError
from app.db.models.document_chunk import DocumentChunk as ChunkRow
from app.db.session import get_async_session_local, initialize_database, reset_engine
from app.services.ingestion.chunking import Chunk, chunk_id
from app.services.rag.pg_index import PgChunkIndex

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL", "").startswith("postgresql"), reason="needs PostgreSQL"
)
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def vector(axis: int) -> list[float]:
    vec = [0.0] * 384
    vec[axis % 384] = 1.0
    return vec


async def write(store, tenant, asset, *, run=1, attempt=1, texts=("one",)):
    chunks = [
        Chunk(chunk_id(tenant, asset, run, i), i, text, page_start=1, page_end=1)
        for i, text in enumerate(texts)
    ]
    return await store.replace_asset(
        tenant_id=tenant,
        asset_id=asset,
        run_number=run,
        attempt=attempt,
        chunks=chunks,
        embeddings=[vector(i) for i in range(len(texts))],
        embedding_model=MODEL,
    )


async def raise_to(store, tenant, asset, run, attempt):
    return await store.raise_fence(
        tenant_id=tenant, asset_id=asset, run_number=run, attempt=attempt
    )


async def rows(tenant, asset):
    async with get_async_session_local()() as db:
        result = await db.execute(
            select(ChunkRow.content, ChunkRow.run_number, ChunkRow.attempt)
            .where(ChunkRow.tenant_id == tenant, ChunkRow.asset_id == asset)
            .order_by(ChunkRow.chunk_index)
        )
        return [tuple(r) for r in result.all()]


@pytest.fixture
async def store():
    await initialize_database()
    yield PgChunkIndex(get_async_session_local())
    await reset_engine()


@pytest.fixture
def ids():
    return str(uuid.uuid4()), f"asset-{uuid.uuid4().hex[:12]}"


async def test_an_asset_without_a_fence_cannot_be_written(store, ids):
    tenant, asset = ids
    with pytest.raises(StaleIndexWriteError):
        await write(store, tenant, asset)
    assert await rows(tenant, asset) == []


async def test_the_current_attempt_writes_and_records_its_attempt(store, ids):
    tenant, asset = ids
    await raise_to(store, tenant, asset, 1, 2)
    assert await write(store, tenant, asset, attempt=2, texts=("a", "b")) == 2
    assert await rows(tenant, asset) == [("a", 1, 2), ("b", 1, 2)]


async def test_a_delayed_old_attempt_cannot_overwrite_the_current_index(store, ids):
    """Attempt 2 indexes and commits; a delayed attempt 1, with different
    chunks, is refused and the index keeps exactly attempt 2's chunks."""
    tenant, asset = ids
    await raise_to(store, tenant, asset, 1, 1)
    await raise_to(store, tenant, asset, 1, 2)
    await write(store, tenant, asset, attempt=2, texts=("current",))

    with pytest.raises(StaleIndexWriteError):
        await write(store, tenant, asset, attempt=1, texts=("stale", "and", "longer"))
    assert await rows(tenant, asset) == [("current", 1, 2)]


async def test_a_late_attempt_cannot_fill_an_empty_index(store, ids):
    """The empty-index case: attempt 2 is dispatched (raising the fence) but
    has not written, or failed before writing. Attempt 1 is still refused,
    though the index holds no newer rows, or none at all."""
    tenant, asset = ids
    await raise_to(store, tenant, asset, 1, 1)
    await raise_to(store, tenant, asset, 1, 2)

    with pytest.raises(StaleIndexWriteError):
        await write(store, tenant, asset, attempt=1, texts=("stale",))
    assert await rows(tenant, asset) == []


async def test_the_fence_only_rises(store, ids):
    tenant, asset = ids
    assert await raise_to(store, tenant, asset, 1, 2) == (1, 2)
    # A duplicate, or a late dispatch of an older attempt, leaves it.
    assert await raise_to(store, tenant, asset, 1, 2) == (1, 2)
    assert await raise_to(store, tenant, asset, 1, 1) == (1, 2)
    assert await raise_to(store, tenant, asset, 1, 3) == (1, 3)
    # A reprocess (a new run) starts at attempt 1 and is above run 1, attempt 3.
    assert await raise_to(store, tenant, asset, 2, 1) == (2, 1)
    assert await raise_to(store, tenant, asset, 1, 9) == (2, 1)


async def test_fences_are_per_tenant_and_asset(store, ids):
    tenant, asset = ids
    other_tenant = str(uuid.uuid4())
    await raise_to(store, tenant, asset, 1, 5)
    assert await raise_to(store, other_tenant, asset, 1, 1) == (1, 1)
    assert await raise_to(store, tenant, asset + "-b", 1, 1) == (1, 1)
    await write(store, other_tenant, asset, attempt=1)  # its own fence admits it


async def test_concurrent_writes_by_two_attempts_serialise_and_only_the_fenced_one_lands(
    store, ids
):
    tenant, asset = ids
    await raise_to(store, tenant, asset, 1, 1)
    await raise_to(store, tenant, asset, 1, 2)
    for _ in range(10):
        results = await asyncio.gather(
            write(store, tenant, asset, attempt=1, texts=("stale",)),
            write(store, tenant, asset, attempt=2, texts=("current",)),
            return_exceptions=True,
        )
        assert isinstance(results[0], StaleIndexWriteError)
        assert results[1] == 1
        assert await rows(tenant, asset) == [("current", 1, 2)]


async def test_a_reader_never_sees_a_partial_or_empty_index(store, ids):
    tenant, asset = ids
    texts = tuple(f"chunk {i}" for i in range(20))
    await raise_to(store, tenant, asset, 1, 1)
    await write(store, tenant, asset, attempt=1, texts=texts)
    stop = asyncio.Event()
    seen: set[int] = set()

    async def reader():
        while not stop.is_set():
            seen.add(len(await rows(tenant, asset)))

    async def rewriter():
        for _ in range(15):
            await write(store, tenant, asset, attempt=1, texts=texts)
        stop.set()

    await asyncio.gather(reader(), rewriter())
    assert seen == {len(texts)}
