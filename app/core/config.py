import secrets
import socket
from typing import Literal
from urllib.parse import urlparse, urlsplit, urlunparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Where a missing, short or known service secret must stop startup (design A.1).
STRICT_ENVIRONMENTS = frozenset({"staging", "production"})
MIN_SERVICE_SECRET_BYTES = 32
# The one backend path AI-service ever reports to (design B.3).
BACKEND_CALLBACK_PATH = "/content/ingestion/callbacks"
DEVELOPMENT_BACKEND_CALLBACK_URL = "http://backend:8000" + BACKEND_CALLBACK_PATH
# Values that have appeared in this repository, the backend's, or the M3
# design's test vectors. None of them may sign anything in staging/production.
KNOWN_DEVELOPMENT_SECRETS = frozenset(
    {
        "dev-secret",
        "build-internal-service-secret",
        "build-ai-dev-secret-key",
        "build-ai-dev-webhook-secret",
        "test-only-backend-to-ai-secret-0123456789abcdef",
        "test-only-ai-to-backend-secret-fedcba9876543210",
    }
)


def _resolve_embedded_hostname(url: str) -> str:
    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname not in {"postgres", "redis"}:
        return url

    try:
        socket.getaddrinfo(hostname, parsed.port or 0)
        return url
    except OSError:
        host = "localhost"
        netloc = host
        if parsed.username:
            auth = parsed.username
            if parsed.password:
                auth = f"{auth}:{parsed.password}"
            netloc = f"{auth}@{netloc}"
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        return urlunparse(parsed._replace(netloc=netloc))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", ".env.prod"],
        extra="ignore",
    )

    APP_NAME: str = "ai-service"
    ENV: str = "development"
    APP_ENVIRONMENT: Literal["development", "testing", "staging", "production"] = "development"

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost/build_ai"

    OPENAI_API_KEY: str | None = None
    # Model behind chat, explain and question generation. "deterministic" is an
    # offline stand-in (no network, no model) for development and tests, such
    # as the two-service boundary suite; it is refused in staging/production.
    MODEL_PROVIDER: Literal["openai", "deterministic"] = "openai"

    VECTOR_DB_TYPE: str = "faiss"

    REDIS_URL: str = "redis://localhost:6379"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # Legacy end-user routes (/auth, /exams, the exam AI routes, ws exam):
    # unmounted unless enabled, and enabling them is refused outside
    # development (design B.4, E.2). Deleted in M4.
    LEGACY_USER_ROUTES_ENABLED: bool = False

    # Used only by the legacy routes; must be a real secret when they are on.
    JWT_SECRET: str = "dev-secret"
    JWT_PRIVATE_KEY: str | None = None
    JWT_PUBLIC_KEY: str | None = None
    JWT_ALGORITHM: str = "HS256"
    JWT_AUDIENCE: str | None = None

    # BUILD Service Signature v1: one secret per direction, each with an
    # optional previous secret accepted (never used to sign) during rotation.
    # No defaults: required in staging/production; random per process in
    # testing; absent in development unless set.
    BACKEND_TO_AI_SECRET: str | None = None
    BACKEND_TO_AI_SECRET_PREVIOUS: str | None = None
    AI_TO_BACKEND_SECRET: str | None = None
    AI_TO_BACKEND_SECRET_PREVIOUS: str | None = None
    # Connect/read timeout (seconds) of the Redis replay guard for signed
    # backend requests. A timeout fails closed (503), never admits the
    # request. Default 1.0; allowed 0.05-5.0.
    SERVICE_REPLAY_TIMEOUT_SECONDS: float = Field(default=1.0, ge=0.05, le=5.0)
    # Where ingestion callbacks go: operator configuration only, never request
    # data. Required (https) in staging/production; development and testing
    # default to the compose backend.
    BACKEND_CALLBACK_URL: str | None = None

    CALLBACK_TIMEOUT_SECONDS: float = 10.0
    CALLBACK_MAX_ATTEMPTS: int = 3
    CALLBACK_INITIAL_BACKOFF_SECONDS: float = 0.5
    CALLBACK_MAX_BACKOFF_SECONDS: float = 5.0
    CALLBACK_JITTER: bool = True

    LOG_LEVEL: str = "INFO"

    def __init__(self, **values):
        super().__init__(**values)
        self.DATABASE_URL = _resolve_embedded_hostname(self.DATABASE_URL)
        self.REDIS_URL = _resolve_embedded_hostname(self.REDIS_URL)

    @model_validator(mode="after")
    def _validate_backend_callback_url(self) -> "Settings":
        strict = self.APP_ENVIRONMENT in STRICT_ENVIRONMENTS
        url = self.BACKEND_CALLBACK_URL
        if not url:
            if strict:
                raise ValueError(
                    f"BACKEND_CALLBACK_URL is required when APP_ENVIRONMENT={self.APP_ENVIRONMENT}"
                )
            self.BACKEND_CALLBACK_URL = DEVELOPMENT_BACKEND_CALLBACK_URL
            return self

        parts = urlsplit(url)
        allowed_schemes = {"https"} if strict else {"http", "https"}
        if parts.scheme not in allowed_schemes:
            raise ValueError(
                f"BACKEND_CALLBACK_URL must use {' or '.join(sorted(allowed_schemes))}"
            )
        if not parts.hostname:
            raise ValueError("BACKEND_CALLBACK_URL must name a host")
        if parts.username is not None or parts.password is not None:
            raise ValueError("BACKEND_CALLBACK_URL must not carry credentials")
        if parts.path != BACKEND_CALLBACK_PATH:
            raise ValueError(f"BACKEND_CALLBACK_URL must target {BACKEND_CALLBACK_PATH}")
        if "?" in url or "#" in url:
            raise ValueError("BACKEND_CALLBACK_URL must not have a query or fragment")
        return self

    @model_validator(mode="after")
    def _validate_service_secrets(self) -> "Settings":
        if self.APP_ENVIRONMENT == "testing":
            # Single-process tests sign and verify with the same settings.
            self.BACKEND_TO_AI_SECRET = self.BACKEND_TO_AI_SECRET or secrets.token_urlsafe(48)
            self.AI_TO_BACKEND_SECRET = self.AI_TO_BACKEND_SECRET or secrets.token_urlsafe(48)

        backend_to_ai = {
            "BACKEND_TO_AI_SECRET": self.BACKEND_TO_AI_SECRET,
            "BACKEND_TO_AI_SECRET_PREVIOUS": self.BACKEND_TO_AI_SECRET_PREVIOUS,
        }
        ai_to_backend = {
            "AI_TO_BACKEND_SECRET": self.AI_TO_BACKEND_SECRET,
            "AI_TO_BACKEND_SECRET_PREVIOUS": self.AI_TO_BACKEND_SECRET_PREVIOUS,
        }
        strict = self.APP_ENVIRONMENT in STRICT_ENVIRONMENTS

        for name, value in {**backend_to_ai, **ai_to_backend}.items():
            if value is None:
                if strict and not name.endswith("_PREVIOUS"):
                    raise ValueError(
                        f"{name} is required when APP_ENVIRONMENT={self.APP_ENVIRONMENT}"
                    )
                continue
            if len(value.encode("utf-8")) < MIN_SERVICE_SECRET_BYTES:
                raise ValueError(f"{name} must be at least {MIN_SERVICE_SECRET_BYTES} bytes")
            if strict and value in KNOWN_DEVELOPMENT_SECRETS:
                raise ValueError(f"{name} is a known development value")

        for current, previous in (
            ("BACKEND_TO_AI_SECRET", "BACKEND_TO_AI_SECRET_PREVIOUS"),
            ("AI_TO_BACKEND_SECRET", "AI_TO_BACKEND_SECRET_PREVIOUS"),
        ):
            values = {**backend_to_ai, **ai_to_backend}
            if values[previous] is not None and values[current] is None:
                raise ValueError(f"{previous} requires {current}")

        # The two directions must never share a secret, including across a rotation.
        shared = {v for v in backend_to_ai.values() if v} & {v for v in ai_to_backend.values() if v}
        if shared:
            raise ValueError("BACKEND_TO_AI and AI_TO_BACKEND secrets must differ")
        return self

    @model_validator(mode="after")
    def _validate_model_provider(self) -> "Settings":
        if self.MODEL_PROVIDER == "deterministic" and self.APP_ENVIRONMENT in STRICT_ENVIRONMENTS:
            raise ValueError(
                f"MODEL_PROVIDER=deterministic is refused when APP_ENVIRONMENT={self.APP_ENVIRONMENT}"
            )
        return self

    @model_validator(mode="after")
    def _validate_legacy_user_routes(self) -> "Settings":
        if not self.LEGACY_USER_ROUTES_ENABLED:
            return self
        if self.APP_ENVIRONMENT != "development":
            raise ValueError(
                "LEGACY_USER_ROUTES_ENABLED is refused when "
                f"APP_ENVIRONMENT={self.APP_ENVIRONMENT}"
            )
        if (
            self.JWT_SECRET in KNOWN_DEVELOPMENT_SECRETS
            or len(self.JWT_SECRET.encode("utf-8")) < MIN_SERVICE_SECRET_BYTES
        ):
            raise ValueError(
                "LEGACY_USER_ROUTES_ENABLED requires JWT_SECRET of at least "
                f"{MIN_SERVICE_SECRET_BYTES} bytes, not a default"
            )
        return self

    @property
    def backend_to_ai_verification_secrets(self) -> tuple[str, ...]:
        """Secrets accepted on inbound backend requests: current, then previous."""
        return tuple(
            s for s in (self.BACKEND_TO_AI_SECRET, self.BACKEND_TO_AI_SECRET_PREVIOUS) if s
        )


settings = Settings()
