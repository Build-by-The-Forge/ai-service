import time
import uuid
from typing import Any

from app.core.execution_context import AITask
from app.services.tasks.transport import TaskTransport, get_task_transport


class TaskDispatcher:
    """
    Dispatcher facade responsible for publishing AITask envelopes to the execution platform.
    Does not execute tasks directly; delegates asynchronously to the transport port.
    """

    def __init__(self, transport: TaskTransport | None = None):
        self._transport = transport

    @property
    def transport(self) -> TaskTransport:
        return self._transport or get_task_transport()

    async def dispatch(self, task: AITask) -> str:
        """Publishes an AITask envelope and returns the transport message ID."""
        return await self.transport.publish(task)

    async def dispatch_ingest(
        self,
        job_id: str,
        asset_id: str,
        tenant_id: str,
        correlation_id: str,
        payload: dict[str, Any],
        run_number: int = 1,
        attempt_count: int = 1,
        external_task_id: str | None = None,
        deadline: float | None = None,
        max_attempts: int = 3,
    ) -> AITask:
        """Helper to construct and dispatch a content ingestion task."""
        task_id = external_task_id or f"task_{uuid.uuid4().hex}"
        execution_id = f"exec_{uuid.uuid4().hex}"

        task = AITask(
            task_id=task_id,
            task_type="content.ingestion",
            execution_id=execution_id,
            tenant_id=tenant_id,
            job_id=job_id,
            asset_id=asset_id,
            run_number=run_number,
            attempt_count=attempt_count,
            correlation_id=correlation_id,
            payload=payload,
            created_at=time.time(),
            deadline=deadline,
            max_attempts=max_attempts,
        )

        await self.dispatch(task)
        return task


_default_dispatcher: TaskDispatcher | None = None


def get_task_dispatcher() -> TaskDispatcher:
    global _default_dispatcher
    if _default_dispatcher is None:
        _default_dispatcher = TaskDispatcher()
    return _default_dispatcher


def set_task_dispatcher(dispatcher: TaskDispatcher) -> None:
    global _default_dispatcher
    _default_dispatcher = dispatcher
