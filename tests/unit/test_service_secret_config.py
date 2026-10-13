"""Service secret configuration (design §A.1, §B.7; acceptance row C26)."""

import pytest
from pydantic import ValidationError

from app.core.config import KNOWN_DEVELOPMENT_SECRETS, Settings

B2A = "b2a-" + "x" * 40
B2A_OLD = "b2a-old-" + "y" * 40
A2B = "a2b-" + "z" * 40
A2B_OLD = "a2b-old-" + "w" * 40


# A valid callback URL, so strict environments fail only on the secret rules.
CALLBACK_URL = "https://api.build.example/content/ingestion/callbacks"


def make(environment, **secrets):
    return Settings(
        _env_file=None,
        APP_ENVIRONMENT=environment,
        BACKEND_CALLBACK_URL=CALLBACK_URL,
        **secrets,
    )


@pytest.fixture(autouse=True)
def no_secret_env(monkeypatch):
    for name in (
        "BACKEND_TO_AI_SECRET",
        "BACKEND_TO_AI_SECRET_PREVIOUS",
        "AI_TO_BACKEND_SECRET",
        "AI_TO_BACKEND_SECRET_PREVIOUS",
    ):
        monkeypatch.delenv(name, raising=False)


# ------------------------------------------------------------ strict environments


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_valid_directional_secrets_start(environment):
    settings = make(
        environment,
        BACKEND_TO_AI_SECRET=B2A,
        BACKEND_TO_AI_SECRET_PREVIOUS=B2A_OLD,
        AI_TO_BACKEND_SECRET=A2B,
        AI_TO_BACKEND_SECRET_PREVIOUS=A2B_OLD,
    )
    assert settings.backend_to_ai_verification_secrets == (B2A, B2A_OLD)
    assert settings.AI_TO_BACKEND_SECRET == A2B


@pytest.mark.parametrize("environment", ["staging", "production"])
@pytest.mark.parametrize("missing", ["BACKEND_TO_AI_SECRET", "AI_TO_BACKEND_SECRET"])
def test_missing_secret_refuses_to_start(environment, missing):
    values = {"BACKEND_TO_AI_SECRET": B2A, "AI_TO_BACKEND_SECRET": A2B}
    del values[missing]
    with pytest.raises(ValidationError, match=f"{missing} is required"):
        make(environment, **values)


@pytest.mark.parametrize("environment", ["staging", "production"])
@pytest.mark.parametrize(
    "name",
    [
        "BACKEND_TO_AI_SECRET",
        "AI_TO_BACKEND_SECRET",
        "BACKEND_TO_AI_SECRET_PREVIOUS",
        "AI_TO_BACKEND_SECRET_PREVIOUS",
    ],
)
def test_short_secret_refuses_to_start(environment, name):
    values = {"BACKEND_TO_AI_SECRET": B2A, "AI_TO_BACKEND_SECRET": A2B, name: "s" * 31}
    with pytest.raises(ValidationError, match="at least 32 bytes"):
        make(environment, **values)


def test_length_is_measured_in_bytes():
    # 16 two-byte characters = 32 bytes: accepted.
    make("production", BACKEND_TO_AI_SECRET="é" * 16, AI_TO_BACKEND_SECRET=A2B)


@pytest.mark.parametrize("environment", ["staging", "production"])
@pytest.mark.parametrize("known", sorted(v for v in KNOWN_DEVELOPMENT_SECRETS if len(v) >= 32))
def test_known_development_value_refuses_to_start(environment, known):
    with pytest.raises(ValidationError, match="known development value"):
        make(environment, BACKEND_TO_AI_SECRET=known, AI_TO_BACKEND_SECRET=A2B)
    with pytest.raises(ValidationError, match="known development value"):
        make(environment, BACKEND_TO_AI_SECRET=B2A, AI_TO_BACKEND_SECRET=known)


def test_short_known_defaults_are_refused_by_length():
    for known in (v for v in KNOWN_DEVELOPMENT_SECRETS if len(v) < 32):
        with pytest.raises(ValidationError, match="at least 32 bytes"):
            make("production", BACKEND_TO_AI_SECRET=known, AI_TO_BACKEND_SECRET=A2B)


# ----------------------------------------------------------------- every environment


@pytest.mark.parametrize("environment", ["development", "testing", "staging", "production"])
def test_equal_directional_secrets_refuse_to_start(environment):
    with pytest.raises(ValidationError, match="must differ"):
        make(environment, BACKEND_TO_AI_SECRET=B2A, AI_TO_BACKEND_SECRET=B2A)


@pytest.mark.parametrize(
    "values",
    [
        {"BACKEND_TO_AI_SECRET_PREVIOUS": A2B},
        {"AI_TO_BACKEND_SECRET_PREVIOUS": B2A},
        {
            "BACKEND_TO_AI_SECRET_PREVIOUS": "shared-" + "q" * 40,
            "AI_TO_BACKEND_SECRET_PREVIOUS": "shared-" + "q" * 40,
        },
    ],
)
def test_directions_cannot_share_a_secret_across_rotation(values):
    with pytest.raises(ValidationError, match="must differ"):
        make("production", BACKEND_TO_AI_SECRET=B2A, AI_TO_BACKEND_SECRET=A2B, **values)


def test_previous_secret_requires_a_current_one():
    with pytest.raises(ValidationError, match="requires BACKEND_TO_AI_SECRET"):
        make("development", BACKEND_TO_AI_SECRET_PREVIOUS=B2A_OLD)


def test_short_secret_is_refused_in_development_too():
    with pytest.raises(ValidationError, match="at least 32 bytes"):
        make("development", BACKEND_TO_AI_SECRET="short")


def test_unknown_environment_is_refused():
    with pytest.raises(ValidationError):
        make("prod")


# ------------------------------------------------------------ non-strict environments


def test_development_has_no_built_in_secrets():
    settings = make("development")
    assert settings.BACKEND_TO_AI_SECRET is None
    assert settings.AI_TO_BACKEND_SECRET is None
    assert settings.backend_to_ai_verification_secrets == ()


def test_development_accepts_known_values_for_local_work():
    make(
        "development",
        BACKEND_TO_AI_SECRET="test-only-backend-to-ai-secret-0123456789abcdef",
        AI_TO_BACKEND_SECRET="test-only-ai-to-backend-secret-fedcba9876543210",
    )


def test_testing_generates_distinct_random_secrets_per_process():
    first, second = make("testing"), make("testing")
    assert first.BACKEND_TO_AI_SECRET != first.AI_TO_BACKEND_SECRET
    assert first.BACKEND_TO_AI_SECRET != second.BACKEND_TO_AI_SECRET
    assert len(first.BACKEND_TO_AI_SECRET.encode()) >= 32


def test_static_bearer_and_legacy_service_settings_are_gone():
    settings = make("development")
    for removed in (
        "SERVICE_AUTH_SECRET",
        "ALLOWED_SERVICE_IDENTITIES",
        "SERVICE_AUTH_MAX_SKEW_SECONDS",
        "AI_SERVICE_API_KEY",
        "AI_SERVICE_WEBHOOK_SECRET",
    ):
        assert not hasattr(settings, removed)


def test_replay_guard_timeout_defaults_to_one_second():
    assert make("development").SERVICE_REPLAY_TIMEOUT_SECONDS == 1.0


@pytest.mark.parametrize("value", [0.0, 0.01, 5.5, -1])
def test_replay_guard_timeout_is_bounded(value):
    with pytest.raises(ValidationError):
        make("development", SERVICE_REPLAY_TIMEOUT_SECONDS=value)
