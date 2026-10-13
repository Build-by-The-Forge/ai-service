import os
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.core import redis as core_redis
from app.core.idempotency import (
    IdempotencyUnavailableError,
    InMemoryIdempotencyStore,
    RedisIdempotencyStore,
)

TEST_REDIS_URL = os.getenv("TEST_REDIS_URL")


@pytest.mark.asyncio
async def test_in_memory_idempotency_store_detects_duplicates():
    store = InMemoryIdempotencyStore()
    unique_key = f"key_{uuid.uuid4().hex}"

    # First attempt should succeed (fresh)
    is_new, existing = await store.set_if_absent(
        tenant_id="school_1",
        idempotency_key=unique_key,
        external_task_id="task_abc",
    )
    assert is_new is True
    assert existing is None

    # Second attempt with same key must return existing record
    is_new2, existing2 = await store.set_if_absent(
        tenant_id="school_1",
        idempotency_key=unique_key,
        external_task_id="task_xyz",
    )
    assert is_new2 is False
    assert existing2 is not None
    assert existing2.external_task_id == "task_abc"

    # Distinct tenant with same key should be treated as separate
    is_new_tenant, _ = await store.set_if_absent(
        tenant_id="school_2",
        idempotency_key=unique_key,
        external_task_id="task_tenant2",
    )
    assert is_new_tenant is True


@pytest.mark.skipif(not TEST_REDIS_URL, reason="TEST_REDIS_URL not set")
@pytest.mark.asyncio
async def test_redis_idempotency_store_operation_and_duplicates(monkeypatch):
    monkeypatch.setattr(core_redis, "redis_url", TEST_REDIS_URL)
    store = RedisIdempotencyStore()
    unique_key = f"key_redis_{uuid.uuid4().hex}"

    is_new, existing = await store.set_if_absent(
        tenant_id="school_1",
        idempotency_key=unique_key,
        external_task_id="task_fb1",
    )
    assert is_new is True
    assert existing is None

    # Duplicate call
    is_new2, existing2 = await store.set_if_absent(
        tenant_id="school_1",
        idempotency_key=unique_key,
        external_task_id="task_fb2",
    )
    assert is_new2 is False
    assert existing2 is not None
    assert existing2.external_task_id == "task_fb1"


@pytest.mark.asyncio
async def test_redis_errors_fail_closed_never_fall_back_to_memory():
    store = RedisIdempotencyStore()
    with patch("app.core.idempotency.get_redis_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.set.side_effect = ConnectionError("Redis unreachable")
        mock_client.get.side_effect = ConnectionError("Redis unreachable")
        mock_get_client.return_value = mock_client

        with pytest.raises(IdempotencyUnavailableError):
            await store.set_if_absent(tenant_id="t", idempotency_key="k", external_task_id="x")
        with pytest.raises(IdempotencyUnavailableError):
            await store.get(tenant_id="t", idempotency_key="k")


@pytest.mark.skipif(not TEST_REDIS_URL, reason="TEST_REDIS_URL not set")
def test_each_event_loop_gets_its_own_working_client(monkeypatch):
    import asyncio

    monkeypatch.setattr(core_redis, "redis_url", TEST_REDIS_URL)

    async def ping():
        return await core_redis.get_redis_client().ping()

    # A second loop used to inherit the first loop's dead connection pool.
    assert asyncio.run(ping()) is True
    assert asyncio.run(ping()) is True
