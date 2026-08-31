import asyncio
import logging
import uuid

from app.core.callbacks.dispatcher import CallbackPort, get_callback_dispatcher
from app.core.callbacks.events import CallbackEvent, CallbackEventType
from app.core.execution_context import AIExecutionContext, AITask, TaskStatus
from app.services.tasks.handlers import TaskHandlerRegistry, get_task_handler_registry
from app.services.tasks.heartbeat import HeartbeatManager, get_heartbeat_manager
from app.services.tasks.transport import TaskTransport, get_task_transport

logger = logging.getLogger(__name__)


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
    ):
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.transport = transport or get_task_transport()
        self.registry = registry or get_task_handler_registry()
        self.heartbeat_mgr = heartbeat_mgr or get_heartbeat_manager()
        self.callback_dispatcher = callback_dispatcher or get_callback_dispatcher()
        self.group_name = group_name
        self._active_contexts: dict[str, AIExecutionContext] = {}

    def get_active_context(self, execution_id: str) -> AIExecutionContext | None:
        return self._active_contexts.get(execution_id)

    def cancel_task(self, execution_id: str, reason: str = "Cancellation requested") -> bool:
        ctx = self._active_contexts.get(execution_id)
        if ctx:
            ctx.cancel(reason)
            return True
        return False

    async def _emit_callback(self, callback_url: str | None, event: CallbackEvent) -> None:
        """
        Emits a callback event safely without allowing delivery exceptions to fail the task execution.
        """
        if not callback_url:
            return
        try:
            result = await self.callback_dispatcher.dispatch(callback_url, event)
            if not result.success:
                logger.warning(
                    "Callback delivery failed: event=%s type=%s status=%s error=%s",
                    event.event_id,
                    event.event_type,
                    result.status_code,
                    result.error_message,
                )
        except Exception as exc:
            logger.error(
                "Unexpected exception during callback dispatch: %s",
                str(exc),
                exc_info=True,
            )

    async def process_single_task(self, msg_id: str, task: AITask) -> TaskStatus:
        callback_url = task.payload.get("callback_url") if isinstance(task.payload, dict) else None

        # 1. Check expiration
        if task.is_expired():
            logger.warning("Task %s is expired (deadline passed). Discarding.", task.task_id)
            await self.transport.ack(msg_id, group=self.group_name)
            return TaskStatus.EXPIRED

        # 2. Establish Execution Context
        ctx = AIExecutionContext.from_task(task)
        self._active_contexts[ctx.execution_id] = ctx

        # 3. Claim execution lease
        self.heartbeat_mgr.record_heartbeat(ctx.execution_id, self.worker_id, lease_seconds=30)

        # 4. Emit STARTED lifecycle event
        await self._emit_callback(
            callback_url,
            CallbackEvent(
                job_id=task.job_id,
                asset_id=task.asset_id,
                tenant_id=task.tenant_id,
                run_number=task.run_number,
                attempt_count=task.attempt_count,
                external_task_id=task.task_id,
                correlation_id=task.correlation_id,
                event_type=CallbackEventType.STARTED,
                progress_pct=0,
            ),
        )

        # 5. Resolve task handler
        handler = self.registry.get(task.task_type)
        if not handler:
            error_msg = f"No handler registered for task_type: '{task.task_type}'"
            logger.error(error_msg)
            await self.transport.send_to_dlq(task, error_reason=error_msg)
            await self.transport.ack(msg_id, group=self.group_name)
            self.heartbeat_mgr.release(ctx.execution_id)
            self._active_contexts.pop(ctx.execution_id, None)

            # Emit terminal FAILED callback
            await self._emit_callback(
                callback_url,
                CallbackEvent(
                    job_id=task.job_id,
                    asset_id=task.asset_id,
                    tenant_id=task.tenant_id,
                    run_number=task.run_number,
                    attempt_count=task.attempt_count,
                    external_task_id=task.task_id,
                    correlation_id=task.correlation_id,
                    event_type=CallbackEventType.FAILED,
                    error_code="UNREGISTERED_HANDLER",
                    error_message=error_msg,
                    retryable=False,
                ),
            )
            return TaskStatus.DEAD_LETTER

        # 6. Execute Handler
        try:
            handler_result = await handler.handle(ctx, task.payload)
            await self.transport.ack(msg_id, group=self.group_name)
            self.heartbeat_mgr.release(ctx.execution_id)
            self._active_contexts.pop(ctx.execution_id, None)

            # Emit COMPLETED callback
            extracted_metadata = None
            chunk_summary = None
            if isinstance(handler_result, dict):
                extracted_metadata = handler_result.get("extracted_metadata")
                chunk_summary = handler_result.get("chunk_summary")

            await self._emit_callback(
                callback_url,
                CallbackEvent(
                    job_id=task.job_id,
                    asset_id=task.asset_id,
                    tenant_id=task.tenant_id,
                    run_number=task.run_number,
                    attempt_count=task.attempt_count,
                    external_task_id=task.task_id,
                    correlation_id=task.correlation_id,
                    event_type=CallbackEventType.COMPLETED,
                    progress_pct=100,
                    extracted_metadata=extracted_metadata,
                    chunk_summary=chunk_summary,
                ),
            )
            return TaskStatus.SUCCEEDED

        except asyncio.CancelledError as ce:
            logger.info("Task %s execution was cancelled: %s", task.task_id, str(ce))
            await self.transport.ack(msg_id, group=self.group_name)
            self.heartbeat_mgr.release(ctx.execution_id)
            self._active_contexts.pop(ctx.execution_id, None)
            return TaskStatus.CANCELLED

        except Exception as exc:
            error_reason = str(exc)
            logger.error(
                "Task %s failed on attempt %d/%d: %s",
                task.task_id,
                task.attempt_count,
                task.max_attempts,
                error_reason,
                exc_info=True,
            )

            # Check retry eligibility
            if task.attempt_count < task.max_attempts:
                # Re-queue next delivery attempt
                retry_task = AITask(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    execution_id=task.execution_id,
                    tenant_id=task.tenant_id,
                    job_id=task.job_id,
                    asset_id=task.asset_id,
                    run_number=task.run_number,
                    attempt_count=task.attempt_count + 1,
                    correlation_id=task.correlation_id,
                    payload=task.payload,
                    created_at=task.created_at,
                    deadline=task.deadline,
                    max_attempts=task.max_attempts,
                )
                await self.transport.publish(retry_task)
                await self.transport.ack(msg_id, group=self.group_name)
                self.heartbeat_mgr.release(ctx.execution_id)
                self._active_contexts.pop(ctx.execution_id, None)
                # Intermediate retry: do NOT emit terminal FAILED event
                return TaskStatus.FAILED
            else:
                # Exceeded max attempts -> Route to DLQ & emit terminal FAILED callback
                await self.transport.send_to_dlq(
                    task,
                    error_reason=f"Exceeded max attempts ({task.max_attempts}): {error_reason}",
                )
                await self.transport.ack(msg_id, group=self.group_name)
                self.heartbeat_mgr.release(ctx.execution_id)
                self._active_contexts.pop(ctx.execution_id, None)

                await self._emit_callback(
                    callback_url,
                    CallbackEvent(
                        job_id=task.job_id,
                        asset_id=task.asset_id,
                        tenant_id=task.tenant_id,
                        run_number=task.run_number,
                        attempt_count=task.attempt_count,
                        external_task_id=task.task_id,
                        correlation_id=task.correlation_id,
                        event_type=CallbackEventType.FAILED,
                        error_code="MAX_ATTEMPTS_EXCEEDED",
                        error_message=error_reason,
                        retryable=False,
                    ),
                )
                return TaskStatus.DEAD_LETTER

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
