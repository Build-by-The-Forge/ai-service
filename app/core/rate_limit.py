"""Rate limiting setup."""

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    limiter: Limiter | None = Limiter(key_func=get_remote_address)
except ImportError:
    # Rate limiting optional in development
    limiter = None
