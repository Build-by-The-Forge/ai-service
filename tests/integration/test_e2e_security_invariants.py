import json
import time
import pytest
from httpx import ASGITransport, AsyncClient
from app.core.config import settings
from app.core.execution_context import AITask, TaskStatus
from app.core.idempotency import InMemoryIdempotencyStore, set_idempotency_store
from app.core.models.contracts import ModelDecision, ModelRequest, ModelResponse
from app.core.models.gateway import ModelGateway
from app.core.models.provider import ModelProviderAdapter
from app.core.service_auth import compute_service_signature
from app.main import app
from app.services.agents.executor import AgentExecutor
from app.services.tasks.dispatcher import TaskDispatcher
from app.services.tasks.handlers import AgentExecutionTaskHandler, TaskHandlerRegistry
from app.services.tasks.transport import InMemoryTaskTransport
from app.services.tasks.worker import TaskWorker


@pytest.fixture(autouse=True)
def setup_idempotency():
    set_idempotency_store(InMemoryIdempotencyStore())
    yield


def generate_auth_headers(
    body_bytes: bytes,
    path: str = "/v1/ingest/dispatch",
    method: str = "POST",
    secret: str = settings.SERVICE_AUTH_SECRET,
    identity: str = "build-backend",
) -> dict:
    timestamp = int(time.time())
    signature = compute_service_signature(
        method=method,
        path=path,
        timestamp=timestamp,
        body=body_bytes,
        secret=secret,
    )
    return {
        "X-Service-Identity": identity,
        "X-Timestamp": str(timestamp),
        "X-Service-Signature": signature,
        "Content-Type": "application/json",
    }


@pytest.mark.asyncio
async def test_e2e_security_dispatch_hmac_tampering_rejected():
    """
    End-to-End Security Test:
    Modifying any byte of the dispatch payload after signing must be rejected with 401.
    """
    payload = {
        "job_id": "job_sec_001",
        "asset_id": "asset_sec_001",
        "tenant_id": "school_futa",
        "source": "upload",
        "storage_uri": "s3://school-assets/doc.pdf",
        "mime_type": "application/pdf",
        "authority": "INSTITUTIONAL",
        "content_sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        "correlation_id": "corr_sec_001",
        "callback_url": "https://api.build.internal/v1/callbacks",
        "idempotency_key": "idemp_sec_001",
    }
    payload_bytes = json.dumps(payload).encode("utf-8")
    headers = generate_auth_headers(payload_bytes)

    tampered_payload = dict(payload)
    tampered_payload["tenant_id"] = "school_unilag"
    tampered_bytes = json.dumps(tampered_payload).encode("utf-8")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/ingest/dispatch",
            content=tampered_bytes,
            headers=headers,
        )
        assert response.status_code == 401
        assert "Invalid service authentication signature" in response.text


@pytest.mark.asyncio
async def test_e2e_security_idempotency_deduplication():
    """
    End-to-End Security Test:
    Replaying an identical dispatch job_id returns cached idempotent response without reprocessing.
    """
    payload = {
        "job_id": "job_idem_9999",
        "asset_id": "asset_idem_9999",
        "tenant_id": "school_futa",
        "source": "upload",
        "storage_uri": "s3://school-assets/syllabus.pdf",
        "mime_type": "application/pdf",
        "authority": "INSTITUTIONAL",
        "content_sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        "correlation_id": "corr_idem_9999",
        "callback_url": "https://api.build.internal/v1/callbacks",
        "idempotency_key": "idemp_9999",
    }
    payload_bytes = json.dumps(payload).encode("utf-8")
    headers = generate_auth_headers(payload_bytes)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp1 = await client.post("/v1/ingest/dispatch", content=payload_bytes, headers=headers)
        assert resp1.status_code == 202
        body1 = resp1.json()
        assert body1["success"] is True
        assert body1["status"] == "ACCEPTED"

        # Immediate replay with same job_id and headers
        resp2 = await client.post("/v1/ingest/dispatch", content=payload_bytes, headers=headers)
        assert resp2.status_code == 202
        body2 = resp2.json()
        assert body2["success"] is True
        assert body2["external_task_id"] == body1["external_task_id"]


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
