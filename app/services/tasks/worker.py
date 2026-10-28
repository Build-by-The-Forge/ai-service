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
from app.core.config import settings
from app.core.execution_context import AIExecutionContext, AITask, TaskStatus
from app.core.redis import get_redis_client
from app.services.tasks.handlers import TaskHandlerRegistry, get_task_handler_registry
from app.services.tasks.heartbeat import HeartbeatManager, get_heartbeat_manager
from app.services.tasks.ingestion_outcomes import (
    PermanentIngestionError,
    PipelineUnavailableError,
    StageReporter,
    TransientIngestionError,
)
from app.services.tasks.reclaimer import INGESTION_TASK_TYPE, StreamReclaimer
from app.services.tasks.task_records import (
    RedisTaskRecordStore,
    TaskRecordStore,
    TaskRecordUnavailableError,
    TaskState,
    WriteResult,
    get_task_record_store,
)
from app.services.tasks.transport import (
    RedisStreamTransport,
    TaskTransport,
    get_task_transport,
)

logger = logging.getLogger(__name__)

# The lease could not be confirmed: the record store is unreachable.
UNCONFIRMED = "unconfirmed"
# A recorded event is retried briefly before the task is fenced.
RECORD_ATTEMPTS = 3
RECORD_RETRY_SECONDS = 0.2


class TaskWorker:
    """
    Asynchronous worker that consumes AITask envelopes, establishes AIExecutionContext,
    executes handlers with heartbeat/lease tracking, manages retries/dead-lettering,
    and reliably emits lifecycle events via the CallbackPort.

    An ingestion task runs under a lease in its task record (P3-6 §7): taken
    at start, renewed every third of TASK_LEASE_SECONDS and by every recorded
    event, cleared by the terminal one. A refused write (the task was
    reclaimed, or its record lost) fences the task: its handler is
    cancelled, and nothing more is recorded or reported for it.
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
        lease_seconds: float | None = None,
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
        self.lease_seconds = lease_seconds or settings.TASK_LEASE_SECONDS
        # Ingestion tasks whose lease this worker holds, those fenced off,
        # and each running handler (so that a fence can cancel it).
        self._leased: set[str] = set()
        self._fenced: set[str] = set()
        self._handlers: dict[str, tuple[AIExecutionContext, asyncio.Future[Any]]] = {}

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
        A fenced task's events are dropped: this worker no longer holds the task.
        """
        if event.external_task_id in self._fenced or not await self._record(event):
            logger.info(
                "Event dropped for a fenced task: task=%s type=%s",
                event.external_task_id,
                event.event_type,
            )
            return None
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

    async def _record(self, event: CallbackEvent) -> bool:
        """Record the backend body of an ingestion event before delivery, so
        an undelivered event can still be reconciled. Returns False when the
        write is refused, or cannot be confirmed, for a task this worker
        leases: it is then fenced and the event is not delivered. Nothing is
        reported for an ingestion task without proven ownership."""
        task_id = event.external_task_id
        if task_id not in self._recorded_tasks:
            return True
        try:
            body = backend_callback_body(event)
        except Exception:
            body = None
        if body is None:
            return True
        state = {
            CallbackEventType.FAILED: TaskState.FAILED,
            CallbackEventType.COMPLETED: TaskState.SUCCEEDED,
        }.get(event.event_type)
        if task_id not in self._leased:
            # Every ingestion task that runs holds a lease; one that does not
            # has nothing it may report.
            self._fence(task_id, "event without a lease")
            return False
        result = UNCONFIRMED
        for attempt in range(RECORD_ATTEMPTS):
            try:
                result = await self.task_records.record_event(
                    task_id,
                    self.worker_id,
                    event=body,
                    state=state,
                    lease_seconds=self.lease_seconds,
                )
                break
            except TaskRecordUnavailableError as exc:
                logger.warning(
                    "Task record not updated: task=%s event=%s attempt=%d error=%s",
                    task_id,
                    event.event_id,
                    attempt + 1,
                    exc,
                )
                await asyncio.sleep(RECORD_RETRY_SECONDS * (attempt + 1))
        if result != WriteResult.OK:
            self._fence(task_id, f"event not recorded: {result}")
            return False
        if state is not None:  # terminal: nothing more to record for this task
            self._recorded_tasks.discard(task_id)
        return True

    def _fence(self, task_id: str, reason: str) -> None:
        """This worker no longer holds the task: stop its handler before any
        further side effect, and report nothing more for it."""
        if task_id in self._fenced:
            return
        self._fenced.add(task_id)
        logger.warning("Task %s fenced: %s", task_id, reason)
        running = self._handlers.get(task_id)
        if running is not None:
            ctx, handler = running
            ctx.cancel(f"fenced: {reason}")
            handler.cancel()

    async def _take_lease(self, task: AITask) -> str:
        """OK when this worker now holds the task; FENCED, MISSING or
        UNCONFIRMED (store unreachable) otherwise, and then it must not run."""
        try:
            result = await self.task_records.start(task.task_id, self.worker_id, self.lease_seconds)
        except TaskRecordUnavailableError as exc:
            logger.warning("Task lease not taken: task=%s error=%s", task.task_id, exc)
            return UNCONFIRMED
        if result == WriteResult.OK:
            self._leased.add(task.task_id)
        return result

    async def _keep_lease(self, task_id: str) -> None:
        """Renew the lease every third of its length until cancelled. A
        refused renewal fences the task. An unreachable store is retried only
        while the last confirmed lease has a third of its length left; past
        that, another worker may reclaim the task, so it is fenced."""
        loop = asyncio.get_running_loop()
        confirmed = loop.time()  # the lease was taken at or after this
        while True:
            await asyncio.sleep(self.lease_seconds / 3)
            asked = loop.time()
            try:
                result = await self.task_records.renew(task_id, self.worker_id, self.lease_seconds)
            except TaskRecordUnavailableError as exc:
                logger.warning("Task lease not renewed: task=%s error=%s", task_id, exc)
                if loop.time() - confirmed >= self.lease_seconds * 2 / 3:
                    self._fence(task_id, "lease could not be confirmed before it lapses")
                    return
                continue
            if result != WriteResult.OK:
                self._fence(task_id, f"lease renewal refused: {result}")
                return
            confirmed = asked

    async def _release_lease(self, task_id: str) -> bool:
        """Hand the task back before re-queueing it. False if it was no
        longer this worker's (it is then fenced)."""
        if task_id not in self._leased:
            return True
        try:
            result = await self.task_records.release(task_id, self.worker_id)
        except TaskRecordUnavailableError as exc:
            # Not handed back, so no other delivery could take it: stop here.
            # The lapsing lease lets the backend recover the job.
            logger.warning("Task lease not released: task=%s error=%s", task_id, exc)
            self._fence(task_id, "lease release could not be confirmed")
            return False
        if result != WriteResult.OK:
            self._fence(task_id, f"lease release refused: {result}")
            return False
        self._leased.discard(task_id)
        return True

    async def _stop_fenced(self, msg_id: str, ctx: AIExecutionContext) -> TaskStatus:
        # A reclaim may have acknowledged the entry already; XACK repeats harmlessly.
        await self.transport.ack(msg_id, group=self.group_name)
        self._finish(msg_id, ctx)
        return TaskStatus.CANCELLED

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
        try:
            return await self._process(msg_id, task)
        finally:
            self._leased.discard(task.task_id)
            self._fenced.discard(task.task_id)
            self._recorded_tasks.discard(task.task_id)

    async def _process(self, msg_id: str, task: AITask) -> TaskStatus:
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
            taken = await self._take_lease(task)
            if taken != WriteResult.OK:
                # Without proven ownership the task never runs here, and
                # nothing is reported (P3-6 §7.1). Reclaimed, finished, held
                # elsewhere or its record gone: nothing may run this entry, so
                # it is acknowledged. Store unreachable: it stays pending, and
                # the reclaim decides once the store is back.
                logger.warning("Task %s not run: lease %s", task.task_id, taken)
                if taken != UNCONFIRMED:
                    await self.transport.ack(msg_id, group=self.group_name)
                return TaskStatus.CANCELLED
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

        # 6. Execute Handler, renewing the lease while it runs
        running = asyncio.ensure_future(handler.handle(ctx, task.payload))
        self._handlers[task.task_id] = (ctx, running)
        renewal = (
            asyncio.create_task(self._keep_lease(task.task_id))
            if task.task_id in self._leased
            else None
        )
        try:
            handler_result = await running
        except asyncio.CancelledError as ce:
            logger.info("Task %s execution was cancelled: %s", task.task_id, str(ce))
            await self.transport.ack(msg_id, group=self.group_name)
            self._finish(msg_id, ctx)
            return TaskStatus.CANCELLED
        except (PermanentIngestionError, PipelineUnavailableError) as failure:
            if task.task_id in self._fenced:
                return await self._stop_fenced(msg_id, ctx)
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
            if task.task_id in self._fenced:
                return await self._stop_fenced(msg_id, ctx)
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
                # worker's delivery count changes. The lease is handed back
                # first, so that the next delivery can take it.
                if not await self._release_lease(task.task_id):
                    return await self._stop_fenced(msg_id, ctx)
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
        finally:
            if renewal is not None:
                renewal.cancel()
            self._handlers.pop(task.task_id, None)

        if task.task_id in self._fenced:
            return await self._stop_fenced(msg_id, ctx)
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
        if task.task_id in self._fenced:  # completion refused: not this worker's to report
            return TaskStatus.CANCELLED
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

    def _default_reclaimer(self) -> StreamReclaimer | None:
        # Reclaiming needs the real stream and the real records.
        if isinstance(self.transport, RedisStreamTransport) and isinstance(
            self.task_records, RedisTaskRecordStore
        ):
            return StreamReclaimer(
                get_redis_client(),
                self.task_records,
                group=self.group_name,
                lease_seconds=self.lease_seconds,
            )
        return None

    async def _reclaim_loop(self, reclaimer: StreamReclaimer, stop_event: asyncio.Event) -> None:
        """A pass every lease length, beside task processing, so a busy
        worker still reclaims a dead one's entries."""
        while not stop_event.is_set():
            try:
                await reclaimer.run_once()
            except Exception as e:
                logger.warning("Reclaim pass failed: %s", e)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.lease_seconds)
            except TimeoutError:
                pass

    async def start(
        self, stop_event: asyncio.Event, reclaimer: StreamReclaimer | None = None
    ) -> None:
        logger.info("TaskWorker %s started listening on group %s", self.worker_id, self.group_name)
        reclaimer = reclaimer or self._default_reclaimer()
        reclaiming = (
            asyncio.create_task(self._reclaim_loop(reclaimer, stop_event)) if reclaimer else None
        )
        try:
            while not stop_event.is_set():
                try:
                    await self.run_once(block_ms=2000)
                except Exception as e:
                    logger.error("Worker error during loop: %s", str(e), exc_info=True)
                    await asyncio.sleep(1)
        finally:
            if reclaiming is not None:
                reclaiming.cancel()
        logger.info("TaskWorker %s shut down gracefully", self.worker_id)
