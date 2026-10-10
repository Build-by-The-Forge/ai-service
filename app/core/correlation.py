"""The correlation id of the request being served.

For a signed backend request it is the verified context's correlation_id,
which the backend derives from its own request id, so one id follows a user
action across both services: backend logs, AI-service logs and the
X-Correlation-ID response header.
"""

from contextvars import ContextVar

current_correlation_id: ContextVar[str | None] = ContextVar("current_correlation_id", default=None)


def get_correlation_id() -> str | None:
    return current_correlation_id.get()
