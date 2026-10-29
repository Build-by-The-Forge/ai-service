import abc
import json
import time
from dataclasses import asdict, dataclass

from app.core.redis import get_redis_client

PROCESSING = "PROCESSING"
ACCEPTED = "ACCEPTED"


@dataclass
class IdempotencyRecord:
    """One dispatch claim. ``status`` is PROCESSING from the claim until
    the task is queued, then ACCEPTED: only an ACCEPTED claim may be
    answered as a duplicate, so a DUPLICATE always names a queued task.
    ``fingerprint`` identifies the request, so the key cannot be reused for
    different work. Claims written before fingerprints have none and are
    treated as accepted, as they were then."""

    tenant_id: str
    idempotency_key: str
    external_task_id: str
    created_at: float = 0.0
    status: str = PROCESSING
    fingerprint: str | None = None

    @property
    def accepted(self) -> bool:
        return self.fingerprint is None or self.status == ACCEPTED


class IdempotencyUnavailableError(RuntimeError):
    """The idempotency store could not answer. Callers fail closed (503):
    answering from a per-process fallback would let replicas disagree and
    accept the same key twice."""


class IdempotencyStore(abc.ABC):
    """
    Abstract Idempotency Port ensuring the API layer remains decoupled from storage implementation.
    """

    @abc.abstractmethod
    async def get(self, tenant_id: str, idempotency_key: str) -> IdempotencyRecord | None:
        pass

    @abc.abstractmethod
    async def set_if_absent(
        self,
        tenant_id: str,
        idempotency_key: str,
        external_task_id: str,
        ttl_seconds: int = 86400,
        fingerprint: str | None = None,
    ) -> tuple[bool, IdempotencyRecord | None]:
        """
        Attempts to record the idempotency key atomically.
        Returns:
            (True, None) if key was newly set (fresh request).
            (False, existing_record) if key already existed (duplicate request).
        """

    @abc.abstractmethod
    async def mark_accepted(
        self, tenant_id: str, idempotency_key: str, external_task_id: str
    ) -> None:
        """The claim's task is queued: from now on it answers duplicates."""

    @abc.abstractmethod
    async def release(self, tenant_id: str, idempotency_key: str, external_task_id: str) -> None:
        """Remove the claim if it is still this task's: its task was never
        queued, so the same request may claim the key again."""


class InMemoryIdempotencyStore(IdempotencyStore):
    """In-memory idempotency store for testing and fallback operation."""

    def __init__(self):
        self._store: dict[str, IdempotencyRecord] = {}

    def _format_key(self, tenant_id: str, idempotency_key: str) -> str:
        return f"idempotency:{tenant_id}:{idempotency_key}"

    async def get(self, tenant_id: str, idempotency_key: str) -> IdempotencyRecord | None:
        key = self._format_key(tenant_id, idempotency_key)
        return self._store.get(key)

    async def set_if_absent(
        self,
        tenant_id: str,
        idempotency_key: str,
        external_task_id: str,
        ttl_seconds: int = 86400,
        fingerprint: str | None = None,
    ) -> tuple[bool, IdempotencyRecord | None]:
        key = self._format_key(tenant_id, idempotency_key)
        existing = self._store.get(key)
        if existing:
            return False, existing

        record = IdempotencyRecord(
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
            external_task_id=external_task_id,
            created_at=time.time(),
            fingerprint=fingerprint,
        )
        self._store[key] = record
        return True, None

    async def mark_accepted(
        self, tenant_id: str, idempotency_key: str, external_task_id: str
    ) -> None:
        record = self._store.get(self._format_key(tenant_id, idempotency_key))
        if record is not None and record.external_task_id == external_task_id:
            record.status = ACCEPTED

    async def release(self, tenant_id: str, idempotency_key: str, external_task_id: str) -> None:
        key = self._format_key(tenant_id, idempotency_key)
        record = self._store.get(key)
        if record is not None and record.external_task_id == external_task_id:
            del self._store[key]


# Both compare the claim's task id first, so a request can only change
# its own claim, never one another request made after it was released.
_MARK_ACCEPTED = """
local raw = redis.call('GET', KEYS[1])
if not raw then return 0 end
local record = cjson.decode(raw)
if record.external_task_id ~= ARGV[1] then return 0 end
record.status = 'ACCEPTED'
redis.call('SET', KEYS[1], cjson.encode(record), 'KEEPTTL')
return 1
"""
_RELEASE = """
local raw = redis.call('GET', KEYS[1])
if not raw then return 0 end
if cjson.decode(raw).external_task_id ~= ARGV[1] then return 0 end
return redis.call('DEL', KEYS[1])
"""


class RedisIdempotencyStore(IdempotencyStore):
    """Redis-backed atomic idempotency store using SET NX EX. Fails closed:
    any Redis error raises IdempotencyUnavailableError."""

    def _format_key(self, tenant_id: str, idempotency_key: str) -> str:
        return f"idempotency:{tenant_id}:{idempotency_key}"

    async def get(self, tenant_id: str, idempotency_key: str) -> IdempotencyRecord | None:
        key = self._format_key(tenant_id, idempotency_key)
        try:
            data = await get_redis_client().get(key)
        except Exception as exc:
            raise IdempotencyUnavailableError("idempotency store unavailable") from exc
        return IdempotencyRecord(**json.loads(data)) if data else None

    async def set_if_absent(
        self,
        tenant_id: str,
        idempotency_key: str,
        external_task_id: str,
        ttl_seconds: int = 86400,
        fingerprint: str | None = None,
    ) -> tuple[bool, IdempotencyRecord | None]:
        key = self._format_key(tenant_id, idempotency_key)
        record = IdempotencyRecord(
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
            external_task_id=external_task_id,
            created_at=time.time(),
            fingerprint=fingerprint,
        )
        try:
            client = get_redis_client()
            if await client.set(key, json.dumps(asdict(record)), ex=ttl_seconds, nx=True):
                return True, None
            existing = await client.get(key)
        except Exception as exc:
            raise IdempotencyUnavailableError("idempotency store unavailable") from exc
        if existing:
            return False, IdempotencyRecord(**json.loads(existing))
        # The key expired between SET and GET: report it as already taken
        # rather than dispatching a second task for it.
        return False, record

    async def _compare(self, script: str, tenant_id: str, idempotency_key: str, task: str) -> None:
        try:
            await get_redis_client().eval(
                script, 1, self._format_key(tenant_id, idempotency_key), task
            )
        except Exception as exc:
            raise IdempotencyUnavailableError("idempotency store unavailable") from exc

    async def mark_accepted(
        self, tenant_id: str, idempotency_key: str, external_task_id: str
    ) -> None:
        await self._compare(_MARK_ACCEPTED, tenant_id, idempotency_key, external_task_id)

    async def release(self, tenant_id: str, idempotency_key: str, external_task_id: str) -> None:
        await self._compare(_RELEASE, tenant_id, idempotency_key, external_task_id)


# Default store singleton provider
_default_store: IdempotencyStore | None = None


def get_idempotency_store() -> IdempotencyStore:
    global _default_store
    if _default_store is None:
        _default_store = RedisIdempotencyStore()
    return _default_store


def set_idempotency_store(store: IdempotencyStore) -> None:
    """Useful for injecting mock/in-memory stores during tests."""
    global _default_store
    _default_store = store
