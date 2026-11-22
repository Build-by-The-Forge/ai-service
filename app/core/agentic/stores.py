"""Where a run keeps its state: checkpoints, and the content they refer to.

A checkpoint holds digests and references; the arguments and results live in a
ResultStore under keys the checkpoint names. Both are Protocols so a server can
use its own store and a device its local database. The in-memory versions are
the reference and the test double.
"""

from __future__ import annotations

from typing import Any, Protocol

from app.core.agentic.contracts import ExecutionCheckpoint, digest


class CheckpointStore(Protocol):
    async def save(self, checkpoint: ExecutionCheckpoint) -> None: ...

    async def load(self, execution_id: str) -> ExecutionCheckpoint | None: ...


class ResultStore(Protocol):
    async def put(self, key: str, payload: dict[str, Any]) -> str:
        """Store the payload; return its sha256 (of its canonical JSON)."""
        ...

    async def get(self, key: str) -> dict[str, Any] | None: ...


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._items: dict[str, str] = {}
        self.saves = 0

    async def save(self, checkpoint: ExecutionCheckpoint) -> None:
        self.saves += 1
        # Serialised, as a database would hold it: what comes back is a copy.
        self._items[checkpoint.execution_id] = checkpoint.model_dump_json()

    async def load(self, execution_id: str) -> ExecutionCheckpoint | None:
        raw = self._items.get(execution_id)
        return ExecutionCheckpoint.model_validate_json(raw) if raw is not None else None


class InMemoryResultStore:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    async def put(self, key: str, payload: dict[str, Any]) -> str:
        self.items[key] = payload
        return digest(payload)

    async def get(self, key: str) -> dict[str, Any] | None:
        return self.items.get(key)
