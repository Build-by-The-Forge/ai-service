"""Reclaiming a dead worker's stream entries, and only a dead worker's (P3-6 §7.3).

Idle time only nominates candidates: ``XPENDING ... IDLE`` lists entries
pending longer than twice the lease, read-only, changing no ownership. A
healthy handler that runs longer than that keeps its entry pending, so idle
time never decides anything. For each ingestion candidate, the task record's
reclaim script decides atomically: a running task whose lease expired under
the entry's consumer (or a task never started) becomes ``lost`` and its
entry is acknowledged, without being run; a live lease is left alone.

The reclaim is stateless and idempotent, so a reclaimer that dies at any
point leaves a state the next pass (by any worker) completes. Recovery of
the job itself is the backend's: a new attempt with a new task.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

from app.core.metrics import expired_lease_candidates, stream_entries_reclaimed_total
from app.services.tasks.task_records import ReclaimResult, TaskRecordStore
from app.services.tasks.transport import TASK_STREAM

logger = logging.getLogger(__name__)

INGESTION_TASK_TYPE = "content.ingestion"


class StreamReclaimer:
    def __init__(
        self,
        client: Any,
        store: TaskRecordStore,
        *,
        group: str,
        lease_seconds: float,
        stream: str = TASK_STREAM,
        batch: int = 100,
    ) -> None:
        self._client = client
        self._store = store
        self._group = group
        self._stream = stream
        self._batch = batch
        self.min_idle_ms = int(lease_seconds * 2 * 1000)

    async def run_once(self) -> dict[str, int]:
        """One pass over the oldest idle candidates; returns outcome counts."""
        outcomes: Counter[str] = Counter()
        expired_count = 0
        candidates = await self._client.xpending_range(
            self._stream,
            self._group,
            min="-",
            max="+",
            count=self._batch,
            idle=self.min_idle_ms,
        )
        for candidate in candidates:
            entry = candidate["message_id"]
            rows = await self._client.xrange(self._stream, min=entry, max=entry, count=1)
            if not rows:
                # Trimmed from the stream while pending: nothing can run it.
                await self._client.xack(self._stream, self._group, entry)
                outcomes["trimmed"] += 1
                stream_entries_reclaimed_total.labels(outcome="trimmed").inc()
                continue
            task = json.loads(rows[0][1].get("data") or "{}")
            if task.get("task_type") != INGESTION_TASK_TYPE or not task.get("task_id"):
                outcomes["skipped"] += 1  # no task record, so no lease to judge by
                continue
            outcome = await self._store.reclaim(
                task["task_id"],
                entry_id=entry,
                consumer=candidate["consumer"],
                idle_ms=int(candidate["time_since_delivered"]),
                min_idle_ms=self.min_idle_ms,
                stream=self._stream,
                group=self._group,
            )
            outcomes[outcome] += 1
            if outcome != ReclaimResult.LIVE:
                stream_entries_reclaimed_total.labels(outcome=outcome).inc()
            if outcome in {ReclaimResult.LOST, ReclaimResult.ACKED}:
                expired_count += 1
            if outcome != ReclaimResult.LIVE:
                logger.warning(
                    "ai_service_stream_entry_reclaimed",
                    extra={
                        "entry_id": entry,
                        "task_id": task.get("task_id"),
                        "job_id": task.get("job_id"),
                        "run_number": task.get("run_number"),
                        "attempt": task.get("attempt_count"),
                        "consumer": candidate["consumer"],
                        "outcome": outcome,
                    },
                )
        expired_lease_candidates.set(expired_count)
        return dict(outcomes)
