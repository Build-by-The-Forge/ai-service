"""BACKEND_CALLBACK_URL is operator configuration with fixed rules (design B.3)."""

import pytest
from pydantic import ValidationError

from app.core.config import DEVELOPMENT_BACKEND_CALLBACK_URL, Settings

SECRETS = {
    "BACKEND_TO_AI_SECRET": "b2a-" + "x" * 40,
    "AI_TO_BACKEND_SECRET": "a2b-" + "z" * 40,
}


@pytest.fixture(autouse=True)
def no_ambient_callback_url(monkeypatch):
    monkeypatch.delenv("BACKEND_CALLBACK_URL", raising=False)


def make(environment, **values):
    return Settings(_env_file=None, APP_ENVIRONMENT=environment, **SECRETS, **values)


@pytest.mark.parametrize("environment", ["development", "testing"])
def test_development_and_testing_default_to_the_compose_backend(environment):
    assert make(environment).BACKEND_CALLBACK_URL == DEVELOPMENT_BACKEND_CALLBACK_URL
    assert DEVELOPMENT_BACKEND_CALLBACK_URL == "http://backend:8000/content/ingestion/callbacks"


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_strict_environments_require_it(environment):
    with pytest.raises(ValidationError, match="BACKEND_CALLBACK_URL is required"):
        make(environment)


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_strict_environments_accept_an_https_url(environment):
    url = "https://api.build.example:8443/content/ingestion/callbacks"
    assert make(environment, BACKEND_CALLBACK_URL=url).BACKEND_CALLBACK_URL == url


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_strict_environments_refuse_plain_http(environment):
    with pytest.raises(ValidationError, match="must use https"):
        make(environment, BACKEND_CALLBACK_URL="http://backend:8000/content/ingestion/callbacks")


def test_development_accepts_http():
    url = "http://127.0.0.1:8000/content/ingestion/callbacks"
    assert make("development", BACKEND_CALLBACK_URL=url).BACKEND_CALLBACK_URL == url


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("https://api.build.example/content/ingestion/callbacks/", "must target"),
        ("https://api.build.example/content/ingestion/callbacks/x", "must target"),
        ("https://api.build.example/content/jobs/1/complete", "must target"),
        ("https://api.build.example/", "must target"),
        ("https://api.build.example/content/ingestion/callbacks?next=/x", "query or fragment"),
        ("https://api.build.example/content/ingestion/callbacks?", "query or fragment"),
        ("https://api.build.example/content/ingestion/callbacks#frag", "query or fragment"),
        ("https://user:pw@api.build.example/content/ingestion/callbacks", "credentials"),
        ("https://user@api.build.example/content/ingestion/callbacks", "credentials"),
        ("https:///content/ingestion/callbacks", "must name a host"),
        ("ftp://api.build.example/content/ingestion/callbacks", "must use https"),
        ("/content/ingestion/callbacks", "must use https"),
    ],
)
def test_malformed_urls_are_refused(url, message):
    with pytest.raises(ValidationError, match=message):
        make("production", BACKEND_CALLBACK_URL=url)
