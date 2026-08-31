import socket
from urllib.parse import urlparse, urlunparse

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost/build_ai"

    OPENAI_API_KEY: str | None = None

    VECTOR_DB_TYPE: str = "faiss"

    REDIS_URL: str = "redis://localhost:6379"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    JWT_SECRET: str = "dev-secret"
    JWT_PRIVATE_KEY: str | None = None
    JWT_PUBLIC_KEY: str | None = None
    JWT_ALGORITHM: str = "HS256"
    JWT_AUDIENCE: str | None = None

    SERVICE_AUTH_SECRET: str = "build-internal-service-secret"
    ALLOWED_SERVICE_IDENTITIES: list[str] = ["build-backend"]
    SERVICE_AUTH_MAX_SKEW_SECONDS: int = 300

    LOG_LEVEL: str = "INFO"

    def __init__(self, **values):
        super().__init__(**values)
        self.DATABASE_URL = _resolve_embedded_hostname(self.DATABASE_URL)
        self.REDIS_URL = _resolve_embedded_hostname(self.REDIS_URL)


settings = Settings()
