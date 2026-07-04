import socket


from app.core.config import Settings, _resolve_embedded_hostname


def test_resolve_embedded_hostname_returns_same_for_local_host():
    url = "postgresql+asyncpg://postgres:postgres@localhost/build_ai"

    assert _resolve_embedded_hostname(url) == url


def test_resolve_embedded_hostname_replaces_embedded_host_with_localhost(monkeypatch):
    url = "postgresql+asyncpg://postgres:postgres@postgres/build_ai"

    def fake_getaddrinfo(hostname, port):
        raise OSError("could not resolve")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    resolved = _resolve_embedded_hostname(url)

    assert resolved.startswith("postgresql+asyncpg://postgres:postgres@localhost")
    assert "postgres" not in resolved.split("@")[-1].split("/")[0]


def test_settings_resolves_embedded_hosts_when_dns_fails(monkeypatch):
    monkeypatch.setattr(
        "app.core.config.socket.getaddrinfo",
        lambda hostname, port: (_ for _ in ()).throw(OSError("resolve fail")),
    )

    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://postgres:postgres@postgres/build_ai",
        REDIS_URL="redis://redis:6379",
    )

    assert "localhost" in settings.DATABASE_URL
    assert "localhost" in settings.REDIS_URL
