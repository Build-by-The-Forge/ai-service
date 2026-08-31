from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.core.callbacks.dispatcher import (
    HTTPCallbackDispatcher,
    InMemoryCallbackDispatcher,
)
from app.core.callbacks.events import CallbackEvent, CallbackEventType


@pytest.fixture
def sample_event():
    return CallbackEvent(
        job_id="job_cb_1",
        asset_id="asset_cb_1",
        tenant_id="school_1",
        run_number=1,
        attempt_count=1,
        external_task_id="task_cb_1",
        correlation_id="corr_cb_1",
        event_type=CallbackEventType.COMPLETED,
        progress_pct=100,
    )


@pytest.mark.asyncio
async def test_in_memory_callback_dispatcher(sample_event):
    dispatcher = InMemoryCallbackDispatcher()

    result = await dispatcher.dispatch("https://backend.build.edu/cb", sample_event)
    assert result.success is True
    assert result.status_code == 200
    assert len(dispatcher.delivered_events) == 1
    assert dispatcher.delivered_events[0].event_type == CallbackEventType.COMPLETED

    # Simulated failure
    dispatcher.simulate_failure(status_code=503)
    fail_result = await dispatcher.dispatch("https://backend.build.edu/cb", sample_event)
    assert fail_result.success is False
    assert fail_result.status_code == 503
    assert fail_result.retryable is True


@pytest.mark.asyncio
async def test_http_callback_successful_delivery(sample_event):
    dispatcher = HTTPCallbackDispatcher(
        timeout_seconds=5.0,
        max_attempts=3,
        initial_backoff=0.01,
        max_backoff=0.05,
        jitter=False,
    )

    mock_response = httpx.Response(status_code=200, json={"received": True})

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        result = await dispatcher.dispatch("https://backend.build.edu/cb", sample_event)
        assert result.success is True
        assert result.status_code == 200
        assert result.attempts == 1

        # Check signature header was injected
        headers = mock_post.call_args.kwargs["headers"]
        assert "X-Build-Signature" in headers
        assert "X-Timestamp" in headers
        assert headers["X-Correlation-ID"] == "corr_cb_1"


@pytest.mark.asyncio
async def test_http_callback_retries_on_503_and_recovers(sample_event):
    dispatcher = HTTPCallbackDispatcher(
        timeout_seconds=5.0,
        max_attempts=3,
        initial_backoff=0.01,
        max_backoff=0.05,
        jitter=False,
    )

    # First attempt: 503, Second attempt: 200
    res_503 = httpx.Response(status_code=503, text="Service Unavailable")
    res_200 = httpx.Response(status_code=200, json={"status": "ok"})

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [res_503, res_200]

        result = await dispatcher.dispatch("https://backend.build.edu/cb", sample_event)
        assert result.success is True
        assert result.status_code == 200
        assert result.attempts == 2
        assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_http_callback_400_is_not_retryable(sample_event):
    dispatcher = HTTPCallbackDispatcher(
        timeout_seconds=5.0,
        max_attempts=3,
        initial_backoff=0.01,
        max_backoff=0.05,
        jitter=False,
    )

    res_400 = httpx.Response(status_code=400, text="Bad Request: Unknown Job ID")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = res_400

        result = await dispatcher.dispatch("https://backend.build.edu/cb", sample_event)
        assert result.success is False
        assert result.status_code == 400
        assert result.retryable is False
        assert result.attempts == 1  # Did not retry
        assert mock_post.call_count == 1


@pytest.mark.asyncio
async def test_http_callback_401_is_not_retryable(sample_event):
    dispatcher = HTTPCallbackDispatcher(
        timeout_seconds=5.0,
        max_attempts=3,
        initial_backoff=0.01,
        max_backoff=0.05,
        jitter=False,
    )

    res_401 = httpx.Response(status_code=401, text="Unauthorized: Invalid Signature")

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = res_401

        result = await dispatcher.dispatch("https://backend.build.edu/cb", sample_event)
        assert result.success is False
        assert result.status_code == 401
        assert result.retryable is False
        assert result.attempts == 1


@pytest.mark.asyncio
async def test_http_callback_invalid_url_rejected(sample_event):
    dispatcher = HTTPCallbackDispatcher()

    result = await dispatcher.dispatch("invalid-url-scheme", sample_event)
    assert result.success is False
    assert result.error_code == "INVALID_CALLBACK_URL"
    assert result.retryable is False
