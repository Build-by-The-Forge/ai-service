"""Legacy end-user routes stay unmounted unless explicitly enabled in
development (design B.4, B.7, E.2; acceptance row C13). M4 deletes them."""

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from app.core.config import Settings
from app.main import create_app

JWT = "legacy-jwt-" + "k" * 40
CALLBACK_URL = "https://api.build.example/content/ingestion/callbacks"
SECRETS = {"BACKEND_TO_AI_SECRET": "b2a-" + "x" * 40, "AI_TO_BACKEND_SECRET": "a2b-" + "z" * 40}

LEGACY_REQUESTS = [
    ("POST", "/auth/login"),
    ("GET", "/exams/"),
    ("POST", "/api/v1/exams/"),
    ("POST", "/api/v1/ai/exam/start"),
    ("POST", "/api/v1/ai/exam/submit"),
    ("POST", "/api/v1/api/v1/generate"),
]


def make(environment: str, **values) -> Settings:
    return Settings(
        _env_file=None,
        APP_ENVIRONMENT=environment,
        BACKEND_CALLBACK_URL=CALLBACK_URL,
        **SECRETS,
        **values,
    )


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    monkeypatch.delenv("LEGACY_USER_ROUTES_ENABLED", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)


def test_legacy_routes_are_off_by_default():
    assert make("development").LEGACY_USER_ROUTES_ENABLED is False


def test_development_may_enable_them_with_a_real_jwt_secret():
    settings = make("development", LEGACY_USER_ROUTES_ENABLED=True, JWT_SECRET=JWT)
    assert settings.LEGACY_USER_ROUTES_ENABLED is True


@pytest.mark.parametrize("environment", ["testing", "staging", "production"])
def test_enabling_them_outside_development_is_refused(environment):
    with pytest.raises(ValidationError, match="LEGACY_USER_ROUTES_ENABLED is refused"):
        make(environment, LEGACY_USER_ROUTES_ENABLED=True, JWT_SECRET=JWT)


@pytest.mark.parametrize("jwt_secret", ["dev-secret", "short"])
def test_enabling_them_requires_a_real_jwt_secret(jwt_secret):
    with pytest.raises(ValidationError, match="requires JWT_SECRET"):
        make("development", LEGACY_USER_ROUTES_ENABLED=True, JWT_SECRET=jwt_secret)


def routes(app) -> set[str]:
    return {getattr(r, "path", "") for r in app.routes}


async def status_of(app, method: str, path: str) -> int:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://ai") as http:
        return (await http.request(method, path, json={})).status_code


@pytest.mark.parametrize("method,path", LEGACY_REQUESTS)
async def test_legacy_routes_are_absent_by_default(method, path):
    assert await status_of(create_app(), method, path) == 404


def test_the_legacy_exam_websocket_is_absent_by_default():
    assert not any("/ws/" in path for path in routes(create_app()))


def test_enabling_them_mounts_the_legacy_routes():
    mounted = routes(create_app(legacy_user_routes=True))
    assert "/auth/login" in mounted
