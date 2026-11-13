"""No exception's text reaches a log, wherever it is caught.

A message can hold a student passage, a prompt, a provider's response body, a
connection string or a URL. Each path that logs an unexpected exception logs
its type and where it was raised (app.core.safe_log.describe), and attaches no
traceback, whose last line is the message again.
"""

import logging

import httpx
import pytest

import app.api.v1.system as system
from app.core.callbacks.dispatcher import HTTPCallbackDispatcher
from app.core.callbacks.events import CallbackEvent, CallbackEventType
from app.core.config import settings
from app.core.safe_log import describe
from app.core.tools.contracts import ToolContext, ToolDefinition, ToolInvocation
from app.core.tools.registry import ToolRegistry
from app.services.tools.executor import ToolExecutor

MARKER = "student-passage-5b8a"


def assert_logged_without_text(caplog) -> None:
    assert caplog.records, "nothing was logged"
    for record in caplog.records:
        assert MARKER not in record.getMessage()
        assert record.exc_info is None


def test_describe_names_the_type_and_place_never_the_text():
    try:
        raise ValueError(MARKER)
    except ValueError as exc:
        described = describe(exc)

    assert MARKER not in described
    assert described.startswith("ValueError at ")
    assert "test_safe_logging.py" in described
    assert describe(KeyError(MARKER)) == "KeyError"  # never raised: no place


class Leaky(Exception):
    """An exception whose own formatting holds the passage, and fails if used."""

    def __str__(self) -> str:
        raise AssertionError(f"describe called str(): {MARKER}")

    def __repr__(self) -> str:
        raise AssertionError(f"describe called repr(): {MARKER}")

    def __format__(self, spec: str) -> str:
        raise AssertionError(f"describe formatted the exception: {MARKER}")


def test_describe_never_formats_the_exception_or_its_cause():
    try:
        try:
            raise ValueError(MARKER)
        except ValueError as cause:
            raise Leaky(MARKER) from cause
    except Leaky as exc:
        described = describe(exc)

    assert described.startswith("Leaky at test_safe_logging.py:")
    assert MARKER not in described and "ValueError" not in described


@pytest.mark.asyncio
async def test_a_crashing_tool_is_logged_without_its_text(caplog):
    registry = ToolRegistry()

    async def crashing(ctx, args):
        raise RuntimeError(f"could not read: {MARKER}")

    registry.register(
        ToolDefinition(
            name="crash.tool", version="1.0.0", description="x", input_schema={"type": "object"}
        ),
        crashing,
    )
    context = ToolContext(
        tenant_id="t",
        execution_id="e",
        task_id="k",
        correlation_id="c",
        permissions=[],
        capabilities=[],
        denied_assets=frozenset(),
    )

    with caplog.at_level(logging.ERROR):
        result = await ToolExecutor(registry=registry, register_builtins=False).execute(
            ToolInvocation(tool_name="crash.tool", arguments={}, context=context)
        )

    assert result.error_code == "TOOL_EXECUTION_FAILED"
    assert_logged_without_text(caplog)


@pytest.mark.asyncio
async def test_an_unhandled_callback_failure_is_logged_and_reported_without_its_text(
    caplog, monkeypatch
):
    monkeypatch.setattr(
        settings, "BACKEND_CALLBACK_URL", "https://api.build.example/content/ingestion/callbacks"
    )
    monkeypatch.setattr(
        settings, "AI_TO_BACKEND_SECRET", "test-ai-to-backend-secret-for-safe-logging-01"
    )
    monkeypatch.setattr(
        settings, "BACKEND_TO_AI_SECRET", "test-backend-to-ai-secret-for-safe-logging-02"
    )

    def broken(request: httpx.Request) -> httpx.Response:
        raise RuntimeError(f"body was: {MARKER}")

    dispatcher = HTTPCallbackDispatcher(
        max_attempts=2, initial_backoff=0.001, jitter=False, transport=httpx.MockTransport(broken)
    )
    event = CallbackEvent(
        job_id="cjob-1",
        asset_id="asset-1",
        tenant_id="6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c",
        run_number=1,
        attempt_count=1,
        external_task_id="task_abc",
        correlation_id="corr-1",
        event_type=CallbackEventType.COMPLETED,
    )

    with caplog.at_level(logging.WARNING):
        result = await dispatcher.dispatch(event)

    assert (result.success, result.error_code) == (False, "UNHANDLED_ERROR")
    assert MARKER not in (result.error_message or "")
    assert_logged_without_text(caplog)


@pytest.mark.asyncio
async def test_a_failed_readiness_check_is_logged_without_its_text(caplog, monkeypatch):
    def unreachable():
        raise ConnectionError(f"postgresql://build:{MARKER}@db/build")

    monkeypatch.setattr(system, "get_engine", unreachable)
    monkeypatch.setattr(system, "get_redis_client", unreachable)

    with caplog.at_level(logging.ERROR):
        await system.readiness()

    assert_logged_without_text(caplog)
