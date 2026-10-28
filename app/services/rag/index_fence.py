"""The index fence as dispatch sees it (P3-6 §7.5).

Dispatch raises the asset's fence to the dispatched (run, attempt) before the
task is queued, so a late or superseded attempt is refused at the database
whether or not the current attempt has written yet. The durable fence is
PgChunkIndex (ingestion_fences); the in-memory one is for unit tests and
follows the same rule: the fence only ever rises.
"""

from __future__ import annotations

from typing import Protocol


class IndexFencePort(Protocol):
    async def raise_fence(
        self, *, tenant_id: str, asset_id: str, run_number: int, attempt: int
    ) -> tuple[int, int]: ...


class InMemoryIndexFence:
    """For tests only."""

    def __init__(self) -> None:
        self.fences: dict[tuple[str, str], tuple[int, int]] = {}

    async def raise_fence(
        self, *, tenant_id: str, asset_id: str, run_number: int, attempt: int
    ) -> tuple[int, int]:
        key = (tenant_id, asset_id)
        self.fences[key] = max(self.fences.get(key, (run_number, attempt)), (run_number, attempt))
        return self.fences[key]


_default_fence: IndexFencePort | None = None


def get_index_fence() -> IndexFencePort:
    global _default_fence
    if _default_fence is None:
        from app.services.rag.pg_index import PgChunkIndex

        _default_fence = PgChunkIndex()
    return _default_fence


def set_index_fence(fence: IndexFencePort | None) -> None:
    global _default_fence
    _default_fence = fence
