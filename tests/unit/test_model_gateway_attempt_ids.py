"""AI-01 decision P3-4: every attempt is traceable on its own.

A retry is a second attempt under the same request: each gets its own
attempt ID, logged with the request's execution ID (its budget scope) and
correlation ID, so a trace can tell the attempts apart and still group them.
The line carries IDs and counts, never the request's content.
"""

import logging
import re

import pytest

from app.core.models.contracts import (
    ModelDecision,
    ModelInvocationContext,
    ModelMessage,
    ModelPolicy,
    ModelRequest,
    ModelResponse,
    ModelRole,
)
from app.core.models.errors import ProviderUnavailableError
from app.core.models.gateway import ModelGateway
from app.core.models.providers.mock import MockProviderAdapter

MARKER = "student-question-3c9f"
CONTEXT = ModelInvocationContext(
    tenant_id="school_1", execution_id="exec_42", task_id="chat", correlation_id="corr_7"
)
ATTEMPT = re.compile(
    r"Model attempt (att_[0-9a-f]{16}): execution=exec_42 correlation=corr_7 task=chat"
    r" provider=(\w+) attempt=(\d)/(\d)"
)


class FailsOnce(MockProviderAdapter):
    """Unavailable on its first call, then answers."""

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        if not self.call_history:
            self.call_history.append(request)
            raise ProviderUnavailableError("down", provider=self.name)
        return await super().generate(request, decision)


def request() -> ModelRequest:
    return ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content=MARKER)],
        context=CONTEXT,
        policy=ModelPolicy(preferred_provider="a", max_retries=1),
    )


def attempts(caplog) -> list[tuple[str, ...]]:
    return [m.groups() for r in caplog.records if (m := ATTEMPT.search(r.getMessage()))]


@pytest.mark.asyncio
async def test_a_retry_is_a_second_attempt_with_its_own_id_under_the_same_request(
    caplog, monkeypatch
):
    monkeypatch.setattr(
        "app.core.models.resilience.ModelRetryPolicy.compute_backoff", lambda attempt: 0.0
    )
    gateway = ModelGateway(providers=[FailsOnce(name="a")], approved_providers={"a"})

    with caplog.at_level(logging.INFO, logger="app.core.models.gateway"):
        await gateway.generate(request())

    seen = attempts(caplog)
    assert [(provider, number, total) for _, provider, number, total in seen] == [
        ("a", "1", "2"),
        ("a", "2", "2"),
    ]
    assert len({attempt_id for attempt_id, *_ in seen}) == 2
    # The failure names its attempt too, and nothing logged holds the content.
    failure = next(
        r.getMessage() for r in caplog.records if "Model invocation error" in r.getMessage()
    )
    assert f"attempt_id={seen[0][0]}" in failure
    assert all(MARKER not in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_stream_is_one_attempt_with_an_id(caplog):
    gateway = ModelGateway(providers=[MockProviderAdapter(name="a")], approved_providers={"a"})

    with caplog.at_level(logging.INFO, logger="app.core.models.gateway"):
        async for _ in gateway.generate_stream(request()):
            pass

    assert [(provider, number, total) for _, provider, number, total in attempts(caplog)] == [
        ("a", "1", "1")
    ]
