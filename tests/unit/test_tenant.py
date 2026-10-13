import asyncio

import pytest

from app.core import tenant


@pytest.fixture(autouse=True)
def no_scope():
    tenant.clear_request_scope()
    yield
    tenant.clear_request_scope()


def test_request_scope_holds_the_verified_tenant_and_user():
    tenant.set_request_scope("11111111-1111-4111-8111-111111111111", "user-1")
    assert tenant.get_tenant_id() == "11111111-1111-4111-8111-111111111111"
    assert tenant.get_user_id() == "user-1"


def test_reading_an_unbound_scope_fails_closed():
    with pytest.raises(tenant.RequestScopeMissingError):
        tenant.get_tenant_id()
    with pytest.raises(tenant.RequestScopeMissingError):
        tenant.get_user_id()


def test_a_system_scope_has_a_tenant_but_no_user():
    tenant.set_request_scope("11111111-1111-4111-8111-111111111111")
    assert tenant.get_tenant_id() == "11111111-1111-4111-8111-111111111111"
    with pytest.raises(tenant.RequestScopeMissingError):
        tenant.get_user_id()


def test_an_empty_tenant_is_refused():
    with pytest.raises(tenant.RequestScopeMissingError):
        tenant.set_request_scope("")


def test_scopes_do_not_leak_between_concurrent_requests():
    async def request(tenant_id: str) -> str:
        tenant.set_request_scope(tenant_id, "u")
        await asyncio.sleep(0)
        return tenant.get_tenant_id()

    async def both():
        return await asyncio.gather(
            asyncio.create_task(request("tenant-a")), asyncio.create_task(request("tenant-b"))
        )

    assert asyncio.run(both()) == ["tenant-a", "tenant-b"]
    with pytest.raises(tenant.RequestScopeMissingError):
        tenant.get_tenant_id()
