import asyncio
import dataclasses
import logging
import uuid
from typing import Any

from app.core.callbacks.dispatcher import (
    CallbackPort,
    backend_callback_body,
    get_callback_dispatcher,
)
from app.core.callbacks.events import CallbackDeliveryResult, CallbackEvent, CallbackEventType
from app.core.execution_context import AIExecutionContext, AITask, TaskStatus
from app.services.tasks.handlers import TaskHandlerRegistry, get_task_handler_registry
from app.services.tasks.heartbeat import HeartbeatManager, get_heartbeat_manager
from app.services.tasks.ingestion_outcomes import (
    PermanentIngestionError,
    PipelineUnavailableError,
    StageReporter,
    TransientIngestionError,
)
from app.services.tasks.task_records import (
    TaskRecordStore,
    TaskRecordUnavailableError,
    TaskState,
    get_task_record_store,
)
from app.services.tasks.transport import TaskTransport, get_task_transport

logger = logging.getLogger(__name__)

INGESTION_TASK_TYPE = "content.ingestion"


class TaskWorker:
    """
    Asynchronous worker that consumes AITask envelopes, establishes AIExecutionContext,
    executes handlers with heartbeat/lease tracking, manages retries/dead-lettering,
    and reliably emits lifecycle events via the CallbackPort.
    """

    def __init__(
        self,
        worker_id: str | None = None,
        transport: TaskTransport | None = None,
        registry: TaskHandlerRegistry | None = None,
        heartbeat_mgr: HeartbeatManager | None = None,
        callback_dispatcher: CallbackPort | None = None,
        group_name: str = "ai_worker_group",
        task_records: TaskRecordStore | None = None,
    ):
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.transport = transport or get_task_transport()
        self.registry = registry or get_task_handler_registry()
        self.heartbeat_mgr = heartbeat_mgr or get_heartbeat_manager()
        self.callback_dispatcher = callback_dispatcher or get_callback_dispatcher()
        self.group_name = group_name
        self.task_records = task_records or get_task_record_store()
        # Ingestion tasks whose backend-bound events are recorded for
        # reconciliation (M3-09).
        self._recorded_tasks: set[str] = set()
        self._active_contexts: dict[str, AIExecutionContext] = {}

    def get_active_context(self, execution_id: str) -> AIExecutionContext | None:
        return self._active_contexts.get(execution_id)

    def cancel_task(self, execution_id: str, reason: str = "Cancellation requested") -> bool:
        ctx = self._active_contexts.get(execution_id)
        if ctx:
            ctx.cancel(reason)
            return True
        return False

    async def _emit_callback(self, event: CallbackEvent) -> CallbackDeliveryResult | None:
        """
        Emits a callback event safely without allowing delivery exceptions to fail the task execution.
        The callback destination is fixed by configuration and must never be chosen by the payload.
        """
        await self._record(event)
        try:
            result = await self.callback_dispatcher.dispatch(event)
            if not result.success:
                logger.warning(
                    "Callback delivery failed: event=%s type=%s status=%s error=%s",
                    event.event_id,
                    event.event_type,
                    result.status_code,
                    result.error_message,
                )
            return result
        except Exception as exc:
            logger.error(
                "Unexpected exception during callback dispatch: %s",
                str(exc),
                exc_info=True,
            )
            return None

    async def _record(self, event: CallbackEvent) -> None:
        """Record the backend body of an ingestion event before delivery, so
        an undelivered event can still be reconciled. A record-store outage
        is logged; it never fails the task."""
        if event.external_task_id not in self._recorded_tasks:
            return
        try:
            body = backend_callback_body(event)
        except Exception:
            body = None
        if body is None:
            return
        state = {
            CallbackEventType.FAILED: TaskState.FAILED,
            CallbackEventType.COMPLETED: TaskState.SUCCEEDED,
        }.get(event.event_type)
        try:
            await self.task_records.update(event.external_task_id, state=state, event=body)
        except TaskRecordUnavailableError as exc:
            logger.warning(
                "Task record not updated: task=%s event=%s error=%s",
                event.external_task_id,
                event.event_id,
                exc,
            )
        if state is not None:  # terminal: nothing more to record for this task
            self._recorded_tasks.discard(event.external_task_id)

    def _event(self, task: AITask, event_type: CallbackEventType, **fields) -> CallbackEvent:
        # run_number and attempt_count are the backend's attempt identity:
        # copied unchanged from the dispatch, whatever the worker's own
        # delivery count is.
        return CallbackEvent(
            job_id=task.job_id,
            asset_id=task.asset_id,
            tenant_id=task.tenant_id,
            run_number=task.run_number,
            attempt_count=task.attempt_count,
            external_task_id=task.task_id,
            correlation_id=task.correlation_id,
            event_type=event_type,
            **fields,
        )

    def _stage_reporter(self, task: AITask) -> StageReporter:
        async def deliver(stage: str, metrics: dict[str, Any]) -> tuple[bool, bool]:
            result = await self._emit_callback(
                self._event(task, CallbackEventType.STAGE_COMPLETED, stage=stage, metrics=metrics)
            )
            if result is None:
                return False, True
            return result.success, result.retryable

        return StageReporter(deliver)

    def _finish(self, msg_id: str, ctx: AIExecutionContext) -> None:
        self.heartbeat_mgr.release(ctx.execution_id)
        self._active_contexts.pop(ctx.execution_id, None)

    async def _fail_terminally(
        self,
        msg_id: str,
        task: AITask,
        ctx: AIExecutionContext,
        code: str,
        message: str,
        retryable: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        await self.transport.ack(msg_id, group=self.group_name)
        self._finish(msg_id, ctx)
        await self._emit_callback(
            self._event(
                task,
                CallbackEventType.FAILED,
                error_code=code,
                error_message=message,
                retryable=retryable,
                details=details or None,
            )
        )

    async def process_single_task(self, msg_id: str, task: AITask) -> TaskStatus:
        # 1. Check expiration
        if task.is_expired():
            logger.warning("Task %s is expired (deadline passed). Discarding.", task.task_id)
            await self.transport.ack(msg_id, group=self.group_name)
            return TaskStatus.EXPIRED

        # 2. Establish Execution Context
        ctx = AIExecutionContext.from_task(task)
        is_ingestion = task.task_type == INGESTION_TASK_TYPE
        if is_ingestion:
            ctx.stage_reporter = self._stage_reporter(task)
            self._recorded_tasks.add(task.task_id)
            try:
                await self.task_records.update(task.task_id, state=TaskState.RUNNING)
            except TaskRecordUnavailableError as exc:
                logger.warning("Task record not updated: task=%s error=%s", task.task_id, exc)
        self._active_contexts[ctx.execution_id] = ctx

        # 3. Claim execution lease
        self.heartbeat_mgr.record_heartbeat(ctx.execution_id, self.worker_id, lease_seconds=30)

        # 4. Emit STARTED lifecycle event (not reported to the backend)
        await self._emit_callback(self._event(task, CallbackEventType.STARTED, progress_pct=0))

        # 5. Resolve task handler
        handler = self.registry.get(task.task_type)
        if not handler:
            error_msg = f"No handler registered for task_type: '{task.task_type}'"
            logger.error(error_msg)
            await self.transport.send_to_dlq(task, error_reason=error_msg)
            await self._fail_terminally(
                msg_id, task, ctx, "UNREGISTERED_HANDLER", error_msg, retryable=False
            )
            return TaskStatus.DEAD_LETTER

        # 6. Execute Handler
        try:
            handler_result = await handler.handle(ctx, task.payload)
        except asyncio.CancelledError as ce:
            logger.info("Task %s execution was cancelled: %s", task.task_id, str(ce))
            await self.transport.ack(msg_id, group=self.group_name)
            self._finish(msg_id, ctx)
            return TaskStatus.CANCELLED
        except (PermanentIngestionError, PipelineUnavailableError) as failure:
            # Repeating the same work cannot help: no internal retry.
            logger.warning(
                "Task %s failed permanently: %s %s", task.task_id, failure.code, failure.message
            )
            await self._fail_terminally(
                msg_id,
                task,
                ctx,
                failure.code,
                failure.message,
                retryable=False,
                details=failure.details,
            )
            return TaskStatus.FAILED
        except Exception as exc:
            transient = isinstance(exc, TransientIngestionError)
            code = exc.code if isinstance(exc, TransientIngestionError) else "AI_UNCLASSIFIED_ERROR"
            logger.error(
                "Task %s failed on delivery %d/%d: %s",
                task.task_id,
                task.delivery_attempt,
                task.max_attempts,
                exc,
                exc_info=not transient,
            )
            if task.delivery_attempt < task.max_attempts:
                # Internal retry of the same backend attempt: only the
                # worker's delivery count changes.
                await self.transport.publish(
                    dataclasses.replace(task, delivery_attempt=task.delivery_attempt + 1)
                )
                await self.transport.ack(msg_id, group=self.group_name)
                self._finish(msg_id, ctx)
                return TaskStatus.FAILED

            await self.transport.send_to_dlq(
                task,
                error_reason=f"Failed after {task.max_attempts} deliveries: {exc}",
            )
            # A transient failure may succeed on a new backend attempt; an
            # unclassified one is a defect to investigate, not to repeat.
            await self._fail_terminally(
                msg_id,
                task,
                ctx,
                code,
                f"Failed after {task.max_attempts} deliveries: {exc}",
                retryable=transient,
                details=exc.details if isinstance(exc, TransientIngestionError) else None,
            )
            return TaskStatus.DEAD_LETTER

        await self.transport.ack(msg_id, group=self.group_name)
        self._finish(msg_id, ctx)

        if is_ingestion and not (ctx.stage_reporter and ctx.stage_reporter.indexed):
            # Success-shaped return without the indexed stage: never report
            # completion the pipeline did not earn.
            await self._emit_callback(
                self._event(
                    task,
                    CallbackEventType.FAILED,
                    error_code="INGESTION_INCOMPLETE",
                    error_message="the handler finished without reporting the indexed stage",
                    retryable=False,
                )
            )
            return TaskStatus.FAILED

        extracted_metadata = None
        chunk_summary = None
        if isinstance(handler_result, dict):
            extracted_metadata = handler_result.get("extracted_metadata")
            chunk_summary = handler_result.get("chunk_summary")
        await self._emit_callback(
            self._event(
                task,
                CallbackEventType.COMPLETED,
                progress_pct=100,
                extracted_metadata=extracted_metadata,
                chunk_summary=chunk_summary,
            )
        )
        return TaskStatus.SUCCEEDED

    async def run_once(self, block_ms: int = 1000) -> TaskStatus | None:
        entries = await self.transport.consume(
            group=self.group_name,
            consumer=self.worker_id,
            count=1,
            block_ms=block_ms,
        )
        if not entries:
            return None

        msg_id, task = entries[0]
        return await self.process_single_task(msg_id, task)

    async def start(self, stop_event: asyncio.Event) -> None:
        logger.info("TaskWorker %s started listening on group %s", self.worker_id, self.group_name)
        while not stop_event.is_set():
            try:
                await self.run_once(block_ms=2000)
            except Exception as e:
                logger.error("Worker error during loop: %s", str(e), exc_info=True)
                await asyncio.sleep(1)
        logger.info("TaskWorker %s shut down gracefully", self.worker_id)
