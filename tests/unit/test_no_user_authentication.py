"""AI-service authenticates services, never users (plan §6.2, M4).

The legacy /auth login, its JWT settings and the legacy-route flag are gone;
a user reaches AI-service only through the backend's signed requests.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings, settings
from app.main import create_app


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/auth/login"),
        ("POST", "/auth/refresh"),
        ("GET", "/auth/me"),
        ("POST", "/auth/logout"),
        ("GET", "/exams/"),
        ("POST", "/api/v1/exams/"),
        ("POST", "/api/v1/ai/exam/start"),
        ("POST", "/api/v1/ai/exam/submit"),
    ],
)
async def test_no_user_login_or_exam_route_exists(method, path):
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://ai") as http:
        assert (await http.request(method, path, json={})).status_code == 404


def test_no_route_serves_a_websocket():
    assert not any(type(r).__name__ == "APIWebSocketRoute" for r in create_app().routes)


@pytest.mark.parametrize(
    "name",
    [
        "JWT_SECRET",
        "JWT_PUBLIC_KEY",
        "JWT_PRIVATE_KEY",
        "JWT_ALGORITHM",
        "LEGACY_USER_ROUTES_ENABLED",
    ],
)
def test_user_authentication_settings_are_gone(name):
    assert name not in Settings.model_fields
    assert not hasattr(settings, name)


@pytest.mark.parametrize(
    "environment,exposed",
    [("development", True), ("testing", True), ("staging", False), ("production", False)],
)
async def test_docs_and_openapi_only_outside_strict_environments(monkeypatch, environment, exposed):
    monkeypatch.setattr(settings, "APP_ENVIRONMENT", environment)
    async with AsyncClient(transport=ASGITransport(app=create_app()), base_url="http://ai") as http:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert ((await http.get(path)).status_code == 200) is exposed, path
