import abc
import json
import time
from dataclasses import asdict, dataclass
from typing import Optional

from app.core.redis import get_redis_client


@dataclass
class IdempotencyRecord:
    tenant_id: str
    idempotency_key: str
    external_task_id: str
    created_at: float = time.time()
    status: str = "PROCESSING"


class IdempotencyStore(abc.ABC):
    """
    Abstract Idempotency Port ensuring the API layer remains decoupled from storage implementation.
    """

    @abc.abstractmethod
    async def get(self, tenant_id: str, idempotency_key: str) -> Optional[IdempotencyRecord]:
        pass

    @abc.abstractmethod
    async def set_if_absent(
        self,
        tenant_id: str,
        idempotency_key: str,
        external_task_id: str,
        ttl_seconds: int = 86400,
    ) -> tuple[bool, Optional[IdempotencyRecord]]:
        """
        Attempts to record the idempotency key atomically.
        Returns:
            (True, None) if key was newly set (fresh request).
            (False, existing_record) if key already existed (duplicate request).
        """
        pass


class InMemoryIdempotencyStore(IdempotencyStore):
    """In-memory idempotency store for testing and fallback operation."""

    def __init__(self):
        self._store: dict[str, IdempotencyRecord] = {}

    def _format_key(self, tenant_id: str, idempotency_key: str) -> str:
        return f"idempotency:{tenant_id}:{idempotency_key}"

    async def get(self, tenant_id: str, idempotency_key: str) -> Optional[IdempotencyRecord]:
        key = self._format_key(tenant_id, idempotency_key)
        return self._store.get(key)

    async def set_if_absent(
        self,
        tenant_id: str,
        idempotency_key: str,
        external_task_id: str,
        ttl_seconds: int = 86400,
    ) -> tuple[bool, Optional[IdempotencyRecord]]:
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
    """Redis-backed atomic idempotency store using SET NX EX."""

    def __init__(self):
        self._in_memory_fallback = InMemoryIdempotencyStore()

    def _format_key(self, tenant_id: str, idempotency_key: str) -> str:
        return f"idempotency:{tenant_id}:{idempotency_key}"

    async def get(self, tenant_id: str, idempotency_key: str) -> Optional[IdempotencyRecord]:
        key = self._format_key(tenant_id, idempotency_key)
        try:
            client = await get_redis_client()
            data = await client.get(key)
            if not data:
                return None
            parsed = json.loads(data)
            return IdempotencyRecord(**parsed)
        except Exception:
            return await self._in_memory_fallback.get(tenant_id, idempotency_key)

    async def set_if_absent(
        self,
        tenant_id: str,
        idempotency_key: str,
        external_task_id: str,
        ttl_seconds: int = 86400,
    ) -> tuple[bool, Optional[IdempotencyRecord]]:
        key = self._format_key(tenant_id, idempotency_key)
        record = IdempotencyRecord(
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
            external_task_id=external_task_id,
            created_at=time.time(),
        )
        try:
            client = await get_redis_client()
            record_json = json.dumps(asdict(record))
            # SET key value NX EX ttl_seconds
            is_new = await client.set(key, record_json, ex=ttl_seconds, nx=True)
            if is_new:
                return True, None
            # Fetch existing
            existing_data = await client.get(key)
            if existing_data:
                parsed = json.loads(existing_data)
                return False, IdempotencyRecord(**parsed)
            return False, record
        except Exception:
            return await self._in_memory_fallback.set_if_absent(
                tenant_id, idempotency_key, external_task_id, ttl_seconds
            )


# Default store singleton provider
_default_store: Optional[IdempotencyStore] = None


def get_idempotency_store() -> IdempotencyStore:
    global _default_store
    if _default_store is None:
        _default_store = RedisIdempotencyStore()
    return _default_store


def set_idempotency_store(store: IdempotencyStore) -> None:
    """Useful for injecting mock/in-memory stores during tests."""
    global _default_store
    _default_store = store
