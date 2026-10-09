"""Until the ingestion pipeline exists, ingestion fails honestly (M3-08)."""

from app.core.callbacks.events import CallbackEventType
from app.core.execution_context import TaskStatus
from app.services.tasks.handlers import ContentIngestionTaskHandler
from app.services.tasks.ingestion_outcomes import PIPELINE_UNAVAILABLE
from tests.unit.test_worker_callback_integration import kinds, run, task, worker_with


async def test_the_stub_pipeline_fails_honestly():
    worker, transport, dispatcher = worker_with(ContentIngestionTaskHandler())
    msg_id, status = await run(worker, transport, task())
    assert status == TaskStatus.FAILED
    assert kinds(dispatcher) == [
        (CallbackEventType.STARTED, None),
        (CallbackEventType.FAILED, None),
    ]
    failed = dispatcher.delivered_events[-1]
    assert (failed.error_code, failed.retryable) == (PIPELINE_UNAVAILABLE, False)
    assert msg_id in transport._acked
    assert transport._queue.empty()  # not retried internally
