import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.core.idempotency import InMemoryIdempotencyStore, RedisIdempotencyStore


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


@pytest.mark.asyncio
async def test_redis_idempotency_store_operation_and_duplicates():
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
async def test_redis_idempotency_store_fallback_on_redis_exception():
    store = RedisIdempotencyStore()
    unique_key = f"key_err_{uuid.uuid4().hex}"

    # Force redis client to raise an exception
    with patch("app.core.idempotency.get_redis_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.set.side_effect = ConnectionError("Redis unreachable")
        mock_client.get.side_effect = ConnectionError("Redis unreachable")
        mock_get_client.return_value = mock_client

        is_new, existing = await store.set_if_absent(
            tenant_id="school_1",
            idempotency_key=unique_key,
            external_task_id="task_fallback",
        )
        assert is_new is True
        assert existing is None

        # Duplicate call under failure fallback
        is_new2, existing2 = await store.set_if_absent(
            tenant_id="school_1",
            idempotency_key=unique_key,
            external_task_id="task_fallback_duplicate",
        )
        assert is_new2 is False
        assert existing2 is not None
        assert existing2.external_task_id == "task_fallback"
