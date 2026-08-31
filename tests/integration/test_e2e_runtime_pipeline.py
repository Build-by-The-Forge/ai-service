import pytest
from app.core.callbacks.dispatcher import InMemoryCallbackDispatcher
from app.core.callbacks.events import CallbackEventType
from app.core.execution_context import AITask, TaskStatus
from app.core.models.contracts import ModelDecision, ModelRequest, ModelResponse, ModelUsage
from app.core.models.gateway import ModelGateway
from app.core.models.provider import ModelProviderAdapter
from app.core.tools.registry import ToolRegistry
from app.services.agents.executor import AgentExecutor
from app.services.tasks.dispatcher import TaskDispatcher
from app.services.tasks.handlers import AgentExecutionTaskHandler, TaskHandlerRegistry
from app.services.tasks.transport import InMemoryTaskTransport
from app.services.tasks.worker import TaskWorker
from app.services.tools.executor import ToolExecutor


class E2EModelProvider(ModelProviderAdapter):
    """
    Simulated LLM provider that executes multi-step reasoning with tool invocations and final answer.
    """

    def __init__(self):
        self.invocations: list[ModelRequest] = []

    @property
    def name(self) -> str:
        return "e2e_provider"

    def supports_model(self, model: str) -> bool:
        return True

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        self.invocations.append(request)
        step = len(self.invocations)

        if step == 1:
            return ModelResponse(
                content="I will search the curriculum database for CSC201.",
                model=decision.model,
                provider=self.name,
                tool_calls=[
                    {
                        "id": "call_curric_1",
                        "type": "function",
                        "function": {
                            "name": "curriculum.lookup",
                            "arguments": '{"course_code": "CSC201"}',
                        },
                    }
                ],
                usage=ModelUsage(
                    prompt_tokens=25,
                    completion_tokens=15,
                    total_tokens=40,
                    estimated_cost_usd=0.0004,
                ),
            )
        elif step == 2:
            return ModelResponse(
                content="I will draft a question proposal based on the curriculum.",
                model=decision.model,
                provider=self.name,
                tool_calls=[
                    {
                        "id": "call_draft_1",
                        "type": "function",
                        "function": {
                            "name": "draft.question",
                            "arguments": '{"topic": "Data Structures", "difficulty": "MEDIUM"}',
                        },
                    }
                ],
                usage=ModelUsage(
                    prompt_tokens=50,
                    completion_tokens=20,
                    total_tokens=70,
                    estimated_cost_usd=0.0007,
                ),
            )
        else:
            return ModelResponse(
                content="CSC201 syllabus retrieved and Question Proposal created successfully.",
                model=decision.model,
                provider=self.name,
                tool_calls=None,
                usage=ModelUsage(
                    prompt_tokens=80,
                    completion_tokens=30,
                    total_tokens=110,
                    estimated_cost_usd=0.0011,
                ),
            )

    async def generate_stream(self, request: ModelRequest, decision: ModelDecision):
        raise NotImplementedError()


@pytest.mark.asyncio
async def test_e2e_happy_path_agent_pipeline_execution():
    """
    End-to-End Test:
    Dispatch Agent Task -> InMemoryTransport -> TaskWorker -> AgentExecutionTaskHandler -> AgentExecutor -> ModelGateway -> ToolExecutor -> Callbacks.
    """
    transport = InMemoryTaskTransport()
    callback_dispatcher = InMemoryCallbackDispatcher()
    model_provider = E2EModelProvider()
    gateway = ModelGateway(providers=[model_provider])
    tool_registry = ToolRegistry()
    tool_executor = ToolExecutor(registry=tool_registry, register_builtins=False)
    agent_executor = AgentExecutor(
        model_gateway=gateway,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
    )

    handler_registry = TaskHandlerRegistry()
    handler_registry.register(
        "agent.execution", AgentExecutionTaskHandler(agent_executor=agent_executor)
    )

    worker = TaskWorker(
        transport=transport,
        registry=handler_registry,
        callback_dispatcher=callback_dispatcher,
        worker_id="worker-e2e-1",
    )

    dispatcher = TaskDispatcher(transport=transport)
    task = AITask(
        task_id="task_e2e_001",
        task_type="agent.execution",
        execution_id="exec_e2e_001",
        tenant_id="school_e2e_futa",
        job_id="job_e2e_001",
        asset_id="asset_e2e_001",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_e2e_001",
        payload={
            "agent_id": "study.assistant",
            "prompt": "Prepare study materials and a draft question for CSC201.",
            "callback_url": "https://backend.build/v1/callbacks/ingest",
        },
    )

    await dispatcher.dispatch(task)

    status = await worker.run_once(block_ms=500)
    assert status == TaskStatus.SUCCEEDED

    delivered = callback_dispatcher.delivered_events
    assert len(delivered) >= 2

    # Verify STARTED callback
    started_evt = delivered[0]
    assert started_evt.event_type == CallbackEventType.STARTED
    assert started_evt.tenant_id == "school_e2e_futa"
    assert started_evt.correlation_id == "corr_e2e_001"

    # Verify COMPLETED callback
    completed_evt = delivered[-1]
    assert completed_evt.event_type == CallbackEventType.COMPLETED
    assert completed_evt.tenant_id == "school_e2e_futa"
    assert completed_evt.correlation_id == "corr_e2e_001"
    assert completed_evt.progress_pct == 100


@pytest.mark.asyncio
async def test_e2e_cross_service_correlation_and_tracing():
    """
    End-to-End Traceability Test:
    Ensures correlation_id, execution_id, task_id, and tenant_id propagate unaltered.
    """
    transport = InMemoryTaskTransport()
    callback_dispatcher = InMemoryCallbackDispatcher()
    model_provider = E2EModelProvider()
    gateway = ModelGateway(providers=[model_provider])
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

    corr_id = "trace_uuid_99999"
    exec_id = "exec_uuid_88888"
    task_id = "task_uuid_77777"
    tenant_id = "school_trace_futa"

    task = AITask(
        task_id=task_id,
        task_type="agent.execution",
        execution_id=exec_id,
        tenant_id=tenant_id,
        job_id="job_trace_1",
        asset_id="asset_trace_1",
        run_number=1,
        attempt_count=1,
        correlation_id=corr_id,
        payload={
            "agent_id": "study.assistant",
            "prompt": "Verify trace correlation",
            "callback_url": "https://backend.build/v1/callbacks/ingest",
        },
    )

    dispatcher = TaskDispatcher(transport=transport)
    await dispatcher.dispatch(task)
    await worker.run_once(block_ms=500)

    for evt in callback_dispatcher.delivered_events:
        assert evt.correlation_id == corr_id
        assert evt.external_task_id == task_id
        assert evt.tenant_id == tenant_id
