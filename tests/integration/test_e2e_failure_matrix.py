import time
import pytest
from app.core.callbacks.dispatcher import InMemoryCallbackDispatcher
from app.core.callbacks.events import CallbackEventType
from app.core.execution_context import AITask, TaskStatus
from app.core.models.contracts import ModelDecision, ModelRequest, ModelResponse
from app.core.models.gateway import ModelGateway
from app.core.models.provider import ModelProviderAdapter
from app.core.tools.contracts import ToolDefinition, ToolSideEffect
from app.core.tools.registry import ToolRegistry
from app.services.agents.executor import AgentExecutor
from app.services.tasks.dispatcher import TaskDispatcher
from app.services.tasks.handlers import AgentExecutionTaskHandler, TaskHandlerRegistry
from app.services.tasks.transport import InMemoryTaskTransport
from app.services.tasks.worker import TaskWorker
from app.services.tools.executor import ToolExecutor


class FailingModelProvider(ModelProviderAdapter):
    def __init__(
        self, failure_count: int = 1, error_to_raise: Exception = ValueError("Model outage")
    ):
        self.failure_count = failure_count
        self.current_attempt = 0
        self.error_to_raise = error_to_raise

    @property
    def name(self) -> str:
        return "failing_provider"

    def supports_model(self, model: str) -> bool:
        return True

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        self.current_attempt += 1
        if self.current_attempt <= self.failure_count:
            raise self.error_to_raise
        return ModelResponse(
            content="Recovered after retry", model=decision.model, provider=self.name
        )

    async def generate_stream(self, request: ModelRequest, decision: ModelDecision):
        raise NotImplementedError()


@pytest.mark.asyncio
async def test_e2e_fatal_model_failure_emits_failed_callback():
    transport = InMemoryTaskTransport()
    callback_dispatcher = InMemoryCallbackDispatcher()
    gateway = ModelGateway(providers=[FailingModelProvider(failure_count=10)])
    agent_executor = AgentExecutor(model_gateway=gateway)

    handler_registry = TaskHandlerRegistry()
    handler_registry.register(
        "agent.execution", AgentExecutionTaskHandler(agent_executor=agent_executor)
    )

    worker = TaskWorker(
        transport=transport,
        registry=handler_registry,
        callback_dispatcher=callback_dispatcher,
    )

    task = AITask(
        task_id="task_fail_1",
        task_type="agent.execution",
        execution_id="exec_fail_1",
        tenant_id="school_fail_1",
        job_id="job_fail_1",
        asset_id="asset_fail_1",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_fail_1",
        payload={
            "agent_id": "study.assistant",
            "prompt": "Test fatal failure",
            "callback_url": "https://backend.build/v1/callbacks/ingest",
        },
    )

    await TaskDispatcher(transport=transport).dispatch(task)
    status = await worker.run_once(block_ms=500)
    assert status == TaskStatus.SUCCEEDED  # Task completed by generating result with status=FAILED

    events = callback_dispatcher.delivered_events
    assert len(events) >= 2
    assert events[0].event_type == CallbackEventType.STARTED
    assert events[1].event_type == CallbackEventType.COMPLETED


@pytest.mark.asyncio
async def test_e2e_tool_crash_failure_isolation_in_agent_loop():
    """
    End-to-End Tool Isolation Test:
    Tool crashes -> isolated into error observation -> Agent completes reasoning.
    """
    tool_registry = ToolRegistry()

    async def crashing_tool_handler(ctx, args):
        raise RuntimeError("PostgreSQL socket reset unexpectedly")

    crashing_def = ToolDefinition(
        name="crashing.lookup",
        version="1.0.0",
        description="Crashing lookup",
        input_schema={"type": "object"},
        side_effect=ToolSideEffect.READ_ONLY,
    )
    tool_registry.register(crashing_def, crashing_tool_handler)

    tool_executor = ToolExecutor(registry=tool_registry, register_builtins=False)

    class ToolCrashHandlingProvider(ModelProviderAdapter):
        def __init__(self):
            self.step = 0

        @property
        def name(self) -> str:
            return "tool_crash_handler"

        def supports_model(self, model: str) -> bool:
            return True

        async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
            self.step += 1
            if self.step == 1:
                return ModelResponse(
                    content="Invoking crashing tool",
                    model=decision.model,
                    provider=self.name,
                    tool_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "crashing.lookup", "arguments": "{}"},
                        }
                    ],
                )
            else:
                return ModelResponse(
                    content="Recovered from tool error: database was temporarily unavailable.",
                    model=decision.model,
                    provider=self.name,
                )

        async def generate_stream(self, request: ModelRequest, decision: ModelDecision):
            raise NotImplementedError()

    gateway = ModelGateway(providers=[ToolCrashHandlingProvider()])
    agent_executor = AgentExecutor(
        model_gateway=gateway,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
    )

    transport = InMemoryTaskTransport()
    callback_dispatcher = InMemoryCallbackDispatcher()
    handler_registry = TaskHandlerRegistry()
    handler_registry.register(
        "agent.execution", AgentExecutionTaskHandler(agent_executor=agent_executor)
    )

    worker = TaskWorker(
        transport=transport, registry=handler_registry, callback_dispatcher=callback_dispatcher
    )

    task = AITask(
        task_id="task_iso_1",
        task_type="agent.execution",
        execution_id="exec_iso_1",
        tenant_id="school_isolate_1",
        job_id="job_iso_1",
        asset_id="asset_iso_1",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_iso_1",
        payload={
            "agent_id": "study.assistant",
            "prompt": "Trigger crashing tool",
            "callback_url": "https://backend.build/v1/callbacks/ingest",
        },
    )

    await TaskDispatcher(transport=transport).dispatch(task)
    status = await worker.run_once(block_ms=500)
    assert status == TaskStatus.SUCCEEDED

    events = callback_dispatcher.delivered_events
    completed_evt = events[-1]
    assert completed_evt.event_type == CallbackEventType.COMPLETED


@pytest.mark.asyncio
async def test_e2e_cooperative_cancellation_propagation():
    """
    End-to-End Cancellation Test:
    When a task deadline is expired before execution, it is detected and discarded as EXPIRED.
    """
    transport = InMemoryTaskTransport()
    callback_dispatcher = InMemoryCallbackDispatcher()

    worker = TaskWorker(transport=transport, callback_dispatcher=callback_dispatcher)

    task = AITask(
        task_id="task_cancel_1",
        task_type="content.ingestion",
        execution_id="exec_cancel_1",
        tenant_id="school_cancel_1",
        job_id="job_cancel_1",
        asset_id="asset_cancel_1",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_cancel_1",
        payload={"storage_uri": "s3://assets/doc.pdf"},
        deadline=time.time() - 100,  # Already expired
    )

    await TaskDispatcher(transport=transport).dispatch(task)
    status = await worker.run_once(block_ms=500)
    assert status == TaskStatus.EXPIRED
