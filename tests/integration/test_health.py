import pytest


async def test_health_endpoint(client):
    res = await client.get("/api/v1/system/health")
    assert res.status_code == 200
    body = res.json()
    assert body.get("success") is True


@pytest.mark.asyncio
async def test_readiness_endpoint(client):
    res = await client.get("/api/v1/system/ready")
    assert res.status_code == 200
    body = res.json()
    assert "data" in body
