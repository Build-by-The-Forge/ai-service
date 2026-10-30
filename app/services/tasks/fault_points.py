"""Test-only pause points for the P3-6 fault-injection tests (§14).

They let a boundary test hold an ingestion handler at an exact point, then
change the world around it (delete the task record, let recovery run) and
release it, so both orderings of a race are reproduced deterministically.

Guardrails:

- Inert unless INGESTION_FAULT_DIR is set, and configuration refuses that
  setting outside APP_ENVIRONMENT=testing: startup fails. pause_point
  checks the environment again before touching the directory.
- No endpoint, header or request field reaches it. The harness controls
  it only through files in that directory, on a volume that only the test
  topology mounts.
- A point is armed while ``<dir>/<point>.pause`` exists. A handler reaching
  it writes ``<dir>/<point>.<job_id>.reached`` and holds until the harness
  removes the ``.pause`` file. The marker is removed when the hold ends.
- While held it does not yield to cancellation. It models an effect that
  is already past its ownership check (in production, a database write in
  flight), which a fence cannot recall; the fence's cancellation takes
  effect at the caller's next ``ctx.is_cancelled()`` check instead.
- A hold that is never released fails after MAX_HOLD_SECONDS rather than
  pinning a worker forever.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

POLL_SECONDS = 0.05
MAX_HOLD_SECONDS = 600.0
BEFORE_INDEX_WRITE = "before_index_write"
AFTER_INDEX_WRITE = "after_index_write"
_NAME = re.compile(r"[a-z_]{1,64}")


class FaultPointNotReleasedError(RuntimeError):
    """The harness never released a held pause point."""


async def pause_point(name: str, job_id: str) -> None:
    from app.core.config import settings

    root = settings.INGESTION_FAULT_DIR
    if not root:
        return
    if settings.APP_ENVIRONMENT != "testing":
        raise RuntimeError("fault points run only when APP_ENVIRONMENT=testing")
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid fault point name {name!r}")
    gate = Path(root) / f"{name}.pause"
    if not gate.exists():
        return
    reached = Path(root) / f"{name}.{re.sub(r'[^A-Za-z0-9_.-]', '_', job_id)}.reached"
    reached.write_text(f"{time.time()}\n", encoding="utf-8")
    deadline = time.monotonic() + MAX_HOLD_SECONDS
    try:
        while gate.exists():
            if time.monotonic() > deadline:
                raise FaultPointNotReleasedError(
                    f"{name} was not released within {MAX_HOLD_SECONDS} s"
                )
            try:
                await asyncio.sleep(POLL_SECONDS)
            except asyncio.CancelledError:
                # Past the check: the effect proceeds once released.
                task = asyncio.current_task()
                if task is not None:
                    task.uncancel()
    finally:
        reached.unlink(missing_ok=True)
