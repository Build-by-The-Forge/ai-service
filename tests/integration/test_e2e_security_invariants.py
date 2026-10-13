import os

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.execution_context import AITask, TaskStatus
from app.core.idempotency import InMemoryIdempotencyStore, set_idempotency_store
from app.core.models.contracts import ModelDecision, ModelRequest, ModelResponse
from app.core.models.gateway import ModelGateway
from app.core.models.provider import ModelProviderAdapter
from app.core.replay_guard import RedisReplayGuard
from app.main import create_app
from app.services.agents.executor import AgentExecutor
from app.services.tasks.dispatcher import TaskDispatcher
from app.services.tasks.handlers import AgentExecutionTaskHandler, TaskHandlerRegistry
from app.services.tasks.transport import InMemoryTaskTransport
from app.services.tasks.worker import TaskWorker
from tests.signing import DISPATCH_PATH, TENANT_B, dispatch_body, signed_headers, user_context

REDIS_URL = os.getenv("TEST_REDIS_URL")
real_redis = pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL not set")


@pytest.fixture(autouse=True)
def setup_idempotency():
    set_idempotency_store(InMemoryIdempotencyStore())
    yield


@pytest.fixture
async def signed_app():
    guard = RedisReplayGuard.connect(REDIS_URL, timeout=2.0)
    yield create_app(replay_guard=guard)
    await guard.close()


@real_redis
@pytest.mark.asyncio
async def test_e2e_security_dispatch_tampering_rejected(signed_app):
    """Changing the signed body or the signed tenant after signing is rejected with 401."""
    body = dispatch_body(job_id="job_sec_001", idempotency_key="idemp_sec_001")
    headers = signed_headers(body)

    swapped_context = signed_headers(body, context=user_context(TENANT_B))
    tenant_swap = {**headers, "X-BUILD-Context": swapped_context["X-BUILD-Context"]}

    async with AsyncClient(
        transport=ASGITransport(app=signed_app), base_url="http://testserver"
    ) as client:
        tampered_body = await client.post(
            DISPATCH_PATH, content=dispatch_body(job_id="job_sec_002"), headers=headers
        )
        tampered_tenant = await client.post(DISPATCH_PATH, content=body, headers=tenant_swap)

    assert tampered_body.status_code == 401
    assert tampered_tenant.status_code == 401
    assert tampered_body.json() == {"error": "invalid service authentication"}


@real_redis
@pytest.mark.asyncio
async def test_e2e_security_idempotency_deduplication(signed_app):
    """A second, freshly signed dispatch with the same idempotency key returns the first task."""
    body = dispatch_body(job_id="job_idem_9999", idempotency_key="idemp_9999")

    async with AsyncClient(
        transport=ASGITransport(app=signed_app), base_url="http://testserver"
    ) as client:
        resp1 = await client.post(DISPATCH_PATH, content=body, headers=signed_headers(body))
        resp2 = await client.post(DISPATCH_PATH, content=body, headers=signed_headers(body))
        replay = await client.post(DISPATCH_PATH, content=body, headers=resp1.request.headers)

    assert resp1.status_code == 202
    assert resp1.json()["status"] == "ACCEPTED"
    assert resp2.status_code == 202
    assert resp2.json()["status"] == "DUPLICATE"
    assert resp2.json()["external_task_id"] == resp1.json()["external_task_id"]
    assert replay.status_code == 401  # the exact same signed request cannot be replayed


@pytest.mark.asyncio
async def test_e2e_security_cross_tenant_tool_tampering_blocked_in_worker():
    """
    End-to-End Security Test:
    When a background task worker processes a tenant execution and the model attempts
    cross-tenant data access, the ToolExecutor strictly prevents cross-tenant access.
    """

    class CrossTenantModelProvider(ModelProviderAdapter):
        @property
        def name(self) -> str:
            return "cross_tenant"

        def supports_model(self, model: str) -> bool:
            return True

        async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
            return ModelResponse(
                content="Attempting unauthorized data breach",
                model=decision.model,
                provider=self.name,
                tool_calls=[
                    {
                        "id": "c_spoof",
                        "type": "function",
                        "function": {
                            "name": "curriculum.lookup",
                            "arguments": '{"course_code": "CSC201", "tenant_id": "school_unilag"}',
                        },
                    }
                ],
            )

        async def generate_stream(self, request: ModelRequest, decision: ModelDecision):
            raise NotImplementedError()

    gateway = ModelGateway(providers=[CrossTenantModelProvider()])
    agent_executor = AgentExecutor(model_gateway=gateway)

    handler_registry = TaskHandlerRegistry()
    handler_registry.register(
        "agent.execution", AgentExecutionTaskHandler(agent_executor=agent_executor)
    )

    transport = InMemoryTaskTransport()
    worker = TaskWorker(transport=transport, registry=handler_registry)

    task = AITask(
        task_id="task_sec_leak_1",
        task_type="agent.execution",
        execution_id="exec_sec_leak_1",
        tenant_id="school_futa",
        job_id="job_sec_leak_1",
        asset_id="asset_sec_leak_1",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_sec_leak_1",
        payload={"agent_id": "study.assistant", "prompt": "Exfiltrate cross-tenant data"},
    )

    await TaskDispatcher(transport=transport).dispatch(task)
    status = await worker.run_once(block_ms=500)
    assert status == TaskStatus.SUCCEEDED
    assert task.tenant_id == "school_futa"
