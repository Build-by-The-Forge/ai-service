from app.core import tenant


def test_tenant_context_isolated():
    tenant.set_school_id(123)
    assert tenant.get_school_id() == 123
