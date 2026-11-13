"""AI-01 decision P3-1: a model execution failure is a 503 with a fixed body.

Whatever the gateway raised, the backend receives the same answer, which it
already treats as the AI service being unavailable, and nothing the provider
or the request said: no message, no provider text, no prompt or passage.
"""

import json
import logging

import pytest

from app.core.errors import model_gateway_error_handler
from app.core.models.errors import (
    AuthenticationError,
    BudgetExceededError,
    CircuitBreakerOpenError,
    FallbackExhaustedError,
    InvalidRequestError,
    ModelGatewayError,
    ModelTimeoutError,
    NoEligibleProviderError,
    ProviderUnavailableError,
    RateLimitExceededError,
)
from app.main import create_app

MARKER = "student-passage-0b7c"
UNAVAILABLE = {
    "success": False,
    "code": "AI_SERVICE_UNAVAILABLE",
    "message": "AI service unavailable",
    "data": {},
}


class State:
    request_id = "req-1"


class Request:
    state = State()


FAILURES = [
    ProviderUnavailableError(MARKER, provider="openai"),
    ModelTimeoutError(MARKER, provider="openai"),
    RateLimitExceededError(MARKER, provider="openai"),
    CircuitBreakerOpenError(MARKER, provider="openai"),
    NoEligibleProviderError(MARKER, provider="openai"),
    FallbackExhaustedError(MARKER, attempted_providers=["openai"]),
    BudgetExceededError(MARKER, budget_type="max_model_calls", limit=2.0, current=2.0),
    InvalidRequestError(MARKER, provider="openai"),  # the provider refused our request
    AuthenticationError(MARKER, provider="openai"),
    ModelGatewayError(MARKER, provider="openai"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", FAILURES, ids=[type(f).__name__ for f in FAILURES])
async def test_every_model_execution_failure_is_the_same_503(failure, caplog):
    with caplog.at_level(logging.WARNING):
        response = await model_gateway_error_handler(Request(), failure)

    assert response.status_code == 503
    assert json.loads(response.body) == UNAVAILABLE
    assert MARKER.encode() not in response.body
    # The log line has the type and safe fields, never the message.
    [line] = [r.getMessage() for r in caplog.records if "answered 503" in r.getMessage()]
    assert type(failure).__name__ in line and "req-1" in line
    assert MARKER not in line


def test_the_app_answers_gateway_failures_with_the_handler():
    assert create_app().exception_handlers[ModelGatewayError] is model_gateway_error_handler
