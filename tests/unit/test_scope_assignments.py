"""Signed scope assignments (M3-D): only the backend's system context sets a
document's scope, in version order, for its own tenant's assets."""

import json
import os

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.rag.scope import KnowledgeScope
from app.core.replay_guard import RedisReplayGuard
from app.core.service_context import ServiceContext
from app.main import create_app
from app.services.rag.scopes import ScopeStoreUnavailableError
from tests.signing import TENANT_A, TENANT_B, signed_headers, user_context

REDIS_URL = os.getenv("TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL not set")
PATH = "/v1/ingest/scope-assignments"


@pytest.fixture
async def app():
    guard = RedisReplayGuard.connect(REDIS_URL, timeout=2.0)
    yield create_app(replay_guard=guard)
    await guard.close()


def system(tenant_id: str = TENANT_A) -> ServiceContext:
    return ServiceContext.system(tenant_id=tenant_id, correlation_id="scope-1")


async def assign(app, body: dict, context: ServiceContext | None = None, signed: bool = True):
    raw = json.dumps(body).encode()
    headers = (
        signed_headers(raw, context=context or system(), target=PATH)
        if signed
        else {"Content-Type": "application/json"}
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://ai-service") as http:
        return await http.post(PATH, content=raw, headers=headers)


def body(asset: str = "asset-1", scope: str = "institutional", version: int = 1) -> dict:
    return {"asset_id": asset, "scope": scope, "version": version}


async def test_the_backend_assigns_a_scope_and_gets_it_back(app, in_memory_scopes):
    response = await assign(app, body(version=3))
    assert response.status_code == 200, response.text
    assert response.json() == {"asset_id": "asset-1", "scope": "institutional", "version": 3}
    assert in_memory_scopes.retrievable(TENANT_A, "asset-1")


async def test_a_replayed_or_older_assignment_changes_nothing_and_is_acknowledged(
    app, in_memory_scopes
):
    await assign(app, body(scope="personal", version=5))
    for version in (5, 4):
        response = await assign(app, body(scope="institutional", version=version))
        assert response.status_code == 200, response.text
        assert response.json() == {"asset_id": "asset-1", "scope": "personal", "version": 5}
    assert not in_memory_scopes.retrievable(TENANT_A, "asset-1")


async def test_a_higher_version_lowers_an_institutional_document(app, in_memory_scopes):
    await assign(app, body(version=1))
    response = await assign(app, body(scope="unknown", version=2))
    assert response.json()["scope"] == "unknown"
    assert not in_memory_scopes.retrievable(TENANT_A, "asset-1")


async def test_another_tenants_asset_is_refused(app, in_memory_scopes):
    await assign(app, body(), context=system(TENANT_A))
    response = await assign(app, body(version=9), context=system(TENANT_B))
    assert response.status_code == 409
    assert in_memory_scopes.rows == {(TENANT_A, "asset-1"): (KnowledgeScope.INSTITUTIONAL, 1)}


async def test_an_unsigned_assignment_is_401(app, in_memory_scopes):
    response = await assign(app, body(), signed=False)
    assert response.status_code == 401
    assert in_memory_scopes.rows == {}


async def test_a_user_context_cannot_assign_a_scope(app, in_memory_scopes):
    staff = user_context(permissions=("content.job.manage", "content.asset.manage"))
    response = await assign(app, body(), context=staff)
    assert response.status_code == 403
    assert in_memory_scopes.rows == {}


@pytest.mark.parametrize(
    "bad",
    [
        {"asset_id": "asset-1", "version": 1},  # no scope
        body(scope="public"),
        body(scope="INSTITUTIONAL"),
        body(scope=""),
        body(version=0),
        body(version=2**63),
        {"asset_id": "asset-1", "scope": "institutional"},  # no version
        body(asset="a/b"),
        body(asset="a\u0000b"),
        body(asset="x" * 65),
        {**body(), "tenant_id": TENANT_B},
    ],
)
async def test_a_missing_or_malformed_assignment_is_422(app, in_memory_scopes, bad):
    response = await assign(app, bad)
    assert response.status_code == 422, response.text
    assert in_memory_scopes.rows == {}


async def test_an_unavailable_store_is_503(app, in_memory_scopes, monkeypatch):
    async def down(**_):
        raise ScopeStoreUnavailableError("OperationalError")

    monkeypatch.setattr(in_memory_scopes, "assign", down)
    response = await assign(app, body())
    assert response.status_code == 503
