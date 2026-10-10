import abc
import json
import time
from dataclasses import asdict, dataclass

from app.core.redis import get_redis_client


@dataclass
class IdempotencyRecord:
    tenant_id: str
    idempotency_key: str
    external_task_id: str
    created_at: float = 0.0
    status: str = "PROCESSING"


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
    ) -> tuple[bool, IdempotencyRecord | None]:
        """
        Attempts to record the idempotency key atomically.
        Returns:
            (True, None) if key was newly set (fresh request).
            (False, existing_record) if key already existed (duplicate request).
        """


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
        )
        self._store[key] = record
        return True, None


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
    ) -> tuple[bool, IdempotencyRecord | None]:
        key = self._format_key(tenant_id, idempotency_key)
        record = IdempotencyRecord(
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
            external_task_id=external_task_id,
            created_at=time.time(),
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
