"""What AI-service knows about each ingestion task, for status reconciliation
(M3-09, design B.3 "Status reconciliation") and task liveness (P3-6 §7).

One record per external task id: the tenant it belongs to (from the verified
dispatch context), the backend attempt identity, the task's state, and the
exact backend callback bodies the worker produced, in order, each with its
event_id. The backend's reconcile operation replays those bodies through the
same callback rules it applies to delivered callbacks, so a callback that was
never delivered can still be applied, and one that was is a duplicate.

Liveness is a lease held by the worker running the task: ``start`` takes it
(``worker_id``, ``lease_until``), ``renew`` extends it, and a terminal state
clears it. Every write a worker makes is conditioned on still holding the
task (same worker, state ``running``), so a worker whose task was reclaimed
or whose record was lost is refused (``FENCED`` / ``MISSING``) and stops.
``stalled`` is never stored: it is a running task whose lease has lapsed
(``reported_state``).

Records live in Redis only: an in-process fallback would give replicas
different answers. In Redis a record is a hash, written only by atomic Lua
scripts that read the server clock; its events are kept as the JSON text the
worker produced, never decoded or re-encoded by a script. Records expire
after TASK_RECORD_TTL_SECONDS, and every write by the running worker
(renewals included) resets that TTL.
"""

from __future__ import annotations

import abc
import json
import time
from dataclasses import dataclass, field
from typing import Any

import redis.asyncio as redis_asyncio
from redis.exceptions import RedisError

TASK_RECORD_TTL_SECONDS = 14 * 24 * 3600
MAX_RECORDED_EVENTS = 20
# The task record and the task stream share this hash tag, so one script
# can use both in a Redis Cluster (P3-6 §7.3).
TASK_KEY_TAG = "{ai-tasks}"


class TaskState:
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    # Reclaimed from a worker whose lease expired; never run again.
    LOST = "lost"
    # Reported, never stored: running with an expired lease.
    STALLED = "stalled"

    TERMINAL = frozenset({SUCCEEDED, FAILED, LOST})


class WriteResult:
    OK = "ok"
    # The record exists, but the caller no longer runs the task.
    FENCED = "fenced"
    MISSING = "missing"


class ReclaimResult:
    # Running under the entry's consumer with an expired lease, or never
    # started: now lost, and the entry acknowledged.
    LOST = "lost"
    # Already lost for this entry, or the record is gone: acknowledged only.
    ACKED = "acked"
    GONE = "gone"
    # A live lease, another owner or another terminal state: untouched.
    LIVE = "live"


class TaskRecordUnavailableError(Exception):
    """The record store could not be reached."""


@dataclass
class TaskRecord:
    external_task_id: str
    tenant_id: str
    job_id: str
    run_number: int
    attempt_count: int
    state: str = TaskState.ACCEPTED
    events: list[dict[str, Any]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)
    worker_id: str | None = None
    lease_until: float | None = None
    heartbeat_at: float | None = None
    lost_entry: str | None = None


def reported_state(record: TaskRecord, now: float | None = None) -> str:
    """The state status reports: ``stalled`` for a running task whose lease
    has lapsed, the stored state otherwise."""
    if record.state == TaskState.RUNNING and (
        record.lease_until is None or record.lease_until <= (time.time() if now is None else now)
    ):
        return TaskState.STALLED
    return record.state


class TaskRecordStore(abc.ABC):
    @abc.abstractmethod
    async def create(self, record: TaskRecord) -> None:
        """Store a new record; an existing record for the task is kept."""

    @abc.abstractmethod
    async def get(self, external_task_id: str) -> TaskRecord | None: ...

    @abc.abstractmethod
    async def start(self, external_task_id: str, worker_id: str, lease_seconds: float) -> str:
        """Take the task's lease: an accepted task becomes running under
        ``worker_id``. Any other state is FENCED."""

    @abc.abstractmethod
    async def renew(self, external_task_id: str, worker_id: str, lease_seconds: float) -> str:
        """Extend the lease while ``worker_id`` still runs the task."""

    @abc.abstractmethod
    async def record_event(
        self,
        external_task_id: str,
        worker_id: str,
        *,
        event: dict[str, Any] | None = None,
        state: str | None = None,
        lease_seconds: float = 0,
    ) -> str:
        """Append an event and/or set the state, only while ``worker_id``
        runs the task; the lease is extended, or cleared by a terminal state.
        There is no unconditional write: a record changes only for its
        owner, or through reclaim."""

    @abc.abstractmethod
    async def release(self, external_task_id: str, worker_id: str) -> str:
        """Hand a running task back to ``accepted`` (re-queued for another
        delivery) while ``worker_id`` still runs it."""

    @abc.abstractmethod
    async def reclaim(
        self,
        external_task_id: str,
        *,
        entry_id: str,
        consumer: str,
        idle_ms: int,
        min_idle_ms: int,
        stream: str,
        group: str,
    ) -> str:
        """Decide a pending stream entry's fate, and acknowledge it when it
        must never run, in one atomic step (P3-6 §7.3)."""


class InMemoryTaskRecordStore(TaskRecordStore):
    """For tests only. The same rules as the Redis scripts, on the local
    clock; ``acked`` lists the stream entries ``reclaim`` acknowledged."""

    def __init__(self) -> None:
        self.records: dict[str, TaskRecord] = {}
        self.acked: list[str] = []

    async def create(self, record: TaskRecord) -> None:
        self.records.setdefault(record.external_task_id, record)

    async def get(self, external_task_id: str) -> TaskRecord | None:
        return self.records.get(external_task_id)

    def _held(self, external_task_id: str, worker_id: str) -> tuple[str, TaskRecord | None]:
        record = self.records.get(external_task_id)
        if record is None:
            return WriteResult.MISSING, None
        if record.state != TaskState.RUNNING or record.worker_id != worker_id:
            return WriteResult.FENCED, None
        return WriteResult.OK, record

    async def start(self, external_task_id: str, worker_id: str, lease_seconds: float) -> str:
        record = self.records.get(external_task_id)
        if record is None:
            return WriteResult.MISSING
        if record.state != TaskState.ACCEPTED:
            return WriteResult.FENCED
        now = time.time()
        record.state, record.worker_id = TaskState.RUNNING, worker_id
        record.lease_until, record.heartbeat_at, record.updated_at = now + lease_seconds, now, now
        return WriteResult.OK

    async def renew(self, external_task_id: str, worker_id: str, lease_seconds: float) -> str:
        result, record = self._held(external_task_id, worker_id)
        if record is not None:
            now = time.time()
            record.lease_until, record.heartbeat_at = now + lease_seconds, now
        return result

    async def record_event(
        self,
        external_task_id: str,
        worker_id: str,
        *,
        event: dict[str, Any] | None = None,
        state: str | None = None,
        lease_seconds: float = 0,
    ) -> str:
        result, record = self._held(external_task_id, worker_id)
        if record is None:
            return result
        now = time.time()
        if event is not None and len(record.events) < MAX_RECORDED_EVENTS:
            record.events.append(event)
        if state is not None:
            record.state = state
        if record.state in TaskState.TERMINAL:
            record.lease_until = None
        else:
            record.lease_until, record.heartbeat_at = now + lease_seconds, now
        record.updated_at = now
        return result

    async def release(self, external_task_id: str, worker_id: str) -> str:
        result, record = self._held(external_task_id, worker_id)
        if record is not None:
            record.state, record.worker_id, record.lease_until = TaskState.ACCEPTED, None, None
            record.updated_at = time.time()
        return result

    async def reclaim(
        self,
        external_task_id: str,
        *,
        entry_id: str,
        consumer: str,
        idle_ms: int,
        min_idle_ms: int,
        stream: str,
        group: str,
    ) -> str:
        record = self.records.get(external_task_id)
        if record is None:
            self.acked.append(entry_id)
            return ReclaimResult.GONE
        if record.state == TaskState.LOST and record.lost_entry == entry_id:
            self.acked.append(entry_id)
            return ReclaimResult.ACKED
        now = time.time()
        expired = (
            record.state == TaskState.RUNNING
            and record.worker_id == consumer
            and record.lease_until is not None
            and record.lease_until < now
        )
        never_started = record.state == TaskState.ACCEPTED and idle_ms >= min_idle_ms
        if not (expired or never_started):
            return ReclaimResult.LIVE
        record.state, record.lost_entry, record.updated_at = TaskState.LOST, entry_id, now
        self.acked.append(entry_id)
        return ReclaimResult.LOST


# Shared by the scripts: the server clock (so workers' clocks never matter),
# and the "this worker runs the task" check.
_LUA_PRELUDE = """
local function now()
  local t = redis.call('TIME')
  return tonumber(t[1]) + tonumber(t[2]) / 1000000
end
local function held(key, worker)
  local f = redis.call('HMGET', key, 'state', 'worker_id')
  if not f[1] then return 'missing' end
  if f[1] ~= 'running' or f[2] ~= worker then return 'fenced' end
  return 'ok'
end
"""

_CREATE = """
if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
redis.call('HSET', KEYS[1], unpack(ARGV, 2))
redis.call('EXPIRE', KEYS[1], ARGV[1])
return 1
"""

# ARGV: worker, lease seconds, TTL.
_START = _LUA_PRELUDE + """
local state = redis.call('HGET', KEYS[1], 'state')
if not state then return 'missing' end
if state ~= 'accepted' then return 'fenced' end
local t = now()
redis.call('HSET', KEYS[1], 'state', 'running', 'worker_id', ARGV[1],
  'lease_until', tostring(t + tonumber(ARGV[2])), 'heartbeat_at', tostring(t),
  'updated_at', tostring(t))
redis.call('EXPIRE', KEYS[1], ARGV[3])
return 'ok'
"""

# ARGV: worker, lease seconds, TTL. The TTL is reset too, so a record can
# never expire under a renewing handler (P3-6 §7.1).
_RENEW = _LUA_PRELUDE + """
local r = held(KEYS[1], ARGV[1])
if r ~= 'ok' then return r end
local t = now()
redis.call('HSET', KEYS[1], 'lease_until', tostring(t + tonumber(ARGV[2])),
  'heartbeat_at', tostring(t))
redis.call('EXPIRE', KEYS[1], ARGV[3])
return 'ok'
"""

# ARGV: worker, event JSON ('' = none), state ('' =
# unchanged), lease seconds, TTL, the event bound. The event text is
# spliced into the array, never decoded.
_RECORD_EVENT = _LUA_PRELUDE + """
local r = held(KEYS[1], ARGV[1])
if r ~= 'ok' then return r end
local t = now()
if ARGV[2] ~= '' then
  local count = tonumber(redis.call('HGET', KEYS[1], 'event_count') or '0')
  if count < tonumber(ARGV[6]) then
    local events
    if count == 0 then
      events = '[' .. ARGV[2] .. ']'
    else
      events = string.sub(redis.call('HGET', KEYS[1], 'events'), 1, -2) .. ',' .. ARGV[2] .. ']'
    end
    redis.call('HSET', KEYS[1], 'events', events, 'event_count', tostring(count + 1))
  end
end
if ARGV[3] ~= '' then redis.call('HSET', KEYS[1], 'state', ARGV[3]) end
local state = redis.call('HGET', KEYS[1], 'state')
if state == 'succeeded' or state == 'failed' or state == 'lost' then
  redis.call('HDEL', KEYS[1], 'lease_until')
else
  redis.call('HSET', KEYS[1], 'lease_until', tostring(t + tonumber(ARGV[4])),
    'heartbeat_at', tostring(t))
end
redis.call('HSET', KEYS[1], 'updated_at', tostring(t))
redis.call('EXPIRE', KEYS[1], ARGV[5])
return 'ok'
"""

# ARGV: worker, TTL.
_RELEASE = _LUA_PRELUDE + """
local r = held(KEYS[1], ARGV[1])
if r ~= 'ok' then return r end
redis.call('HSET', KEYS[1], 'state', 'accepted', 'updated_at', tostring(now()))
redis.call('HDEL', KEYS[1], 'worker_id', 'lease_until')
redis.call('EXPIRE', KEYS[1], ARGV[2])
return 'ok'
"""

# KEYS: record, stream. ARGV: group, entry id, the entry's consumer, its
# idle ms, the idle threshold. The transition to lost and the XACK happen
# together, and a repeat after either is idempotent (P3-6 §7.3).
_RECLAIM = _LUA_PRELUDE + """
local f = redis.call('HMGET', KEYS[1], 'state', 'worker_id', 'lease_until', 'lost_entry')
if not f[1] then
  redis.call('XACK', KEYS[2], ARGV[1], ARGV[2])
  return 'gone'
end
if f[1] == 'lost' and f[4] == ARGV[2] then
  redis.call('XACK', KEYS[2], ARGV[1], ARGV[2])
  return 'acked'
end
local t = now()
local expired = f[1] == 'running' and f[2] == ARGV[3] and f[3] and tonumber(f[3]) < t
local never_started = f[1] == 'accepted' and tonumber(ARGV[4]) >= tonumber(ARGV[5])
if not (expired or never_started) then return 'live' end
redis.call('HSET', KEYS[1], 'state', 'lost', 'lost_entry', ARGV[2], 'updated_at', tostring(t))
redis.call('XACK', KEYS[2], ARGV[1], ARGV[2])
return 'lost'
"""


class RedisTaskRecordStore(TaskRecordStore):
    def __init__(self, client: redis_asyncio.Redis) -> None:
        self._client = client
        self._create = client.register_script(_CREATE)
        self._start = client.register_script(_START)
        self._renew = client.register_script(_RENEW)
        self._record_event = client.register_script(_RECORD_EVENT)
        self._release = client.register_script(_RELEASE)
        self._reclaim = client.register_script(_RECLAIM)

    @classmethod
    def connect(cls, url: str, *, timeout: float = 1.0) -> RedisTaskRecordStore:
        return cls(
            redis_asyncio.Redis.from_url(
                url, decode_responses=True, socket_timeout=timeout, socket_connect_timeout=timeout
            )
        )

    @staticmethod
    def key(external_task_id: str) -> str:
        return f"{TASK_KEY_TAG}:ingest:task:{external_task_id}"

    async def _run(self, script: Any, keys: list[str], args: list[Any]) -> str:
        try:
            return str(await script(keys=keys, args=args))
        except (RedisError, OSError) as exc:
            raise TaskRecordUnavailableError(type(exc).__name__) from exc

    async def create(self, record: TaskRecord) -> None:
        fields = {
            "external_task_id": record.external_task_id,
            "tenant_id": record.tenant_id,
            "job_id": record.job_id,
            "run_number": record.run_number,
            "attempt_count": record.attempt_count,
            "state": record.state,
            "events": json.dumps(record.events),
            "event_count": len(record.events),
            "updated_at": record.updated_at,
        }
        args: list[Any] = [TASK_RECORD_TTL_SECONDS]
        for name, value in fields.items():
            args += [name, value]
        await self._run(self._create, [self.key(record.external_task_id)], args)

    async def get(self, external_task_id: str) -> TaskRecord | None:
        try:
            raw = await self._client.hgetall(self.key(external_task_id))
        except (RedisError, OSError) as exc:
            raise TaskRecordUnavailableError(type(exc).__name__) from exc
        if not raw:
            return None

        def number(name: str) -> float | None:
            return float(raw[name]) if raw.get(name) else None

        return TaskRecord(
            external_task_id=raw["external_task_id"],
            tenant_id=raw["tenant_id"],
            job_id=raw["job_id"],
            run_number=int(raw["run_number"]),
            attempt_count=int(raw["attempt_count"]),
            state=raw["state"],
            events=json.loads(raw.get("events") or "[]"),
            updated_at=float(raw["updated_at"]),
            worker_id=raw.get("worker_id"),
            lease_until=number("lease_until"),
            heartbeat_at=number("heartbeat_at"),
            lost_entry=raw.get("lost_entry"),
        )

    async def start(self, external_task_id: str, worker_id: str, lease_seconds: float) -> str:
        return await self._run(
            self._start,
            [self.key(external_task_id)],
            [worker_id, lease_seconds, TASK_RECORD_TTL_SECONDS],
        )

    async def renew(self, external_task_id: str, worker_id: str, lease_seconds: float) -> str:
        return await self._run(
            self._renew,
            [self.key(external_task_id)],
            [worker_id, lease_seconds, TASK_RECORD_TTL_SECONDS],
        )

    async def record_event(
        self,
        external_task_id: str,
        worker_id: str,
        *,
        event: dict[str, Any] | None = None,
        state: str | None = None,
        lease_seconds: float = 0,
    ) -> str:
        return await self._run(
            self._record_event,
            [self.key(external_task_id)],
            [
                worker_id,
                json.dumps(event) if event is not None else "",
                state or "",
                lease_seconds,
                TASK_RECORD_TTL_SECONDS,
                MAX_RECORDED_EVENTS,
            ],
        )

    async def release(self, external_task_id: str, worker_id: str) -> str:
        return await self._run(
            self._release, [self.key(external_task_id)], [worker_id, TASK_RECORD_TTL_SECONDS]
        )

    async def reclaim(
        self,
        external_task_id: str,
        *,
        entry_id: str,
        consumer: str,
        idle_ms: int,
        min_idle_ms: int,
        stream: str,
        group: str,
    ) -> str:
        return await self._run(
            self._reclaim,
            [self.key(external_task_id), stream],
            [group, entry_id, consumer, idle_ms, min_idle_ms],
        )

    async def close(self) -> None:
        await self._client.close()  # redis-py 5.0.0 has no aclose()


_default_store: TaskRecordStore | None = None


def get_task_record_store() -> TaskRecordStore:
    global _default_store
    if _default_store is None:
        from app.core.config import settings

        _default_store = RedisTaskRecordStore.connect(settings.REDIS_URL)
    return _default_store


def set_task_record_store(store: TaskRecordStore | None) -> None:
    global _default_store
    _default_store = store
