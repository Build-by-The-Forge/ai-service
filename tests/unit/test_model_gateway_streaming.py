import asyncio

import pytest

from app.core.models.contracts import (
    ModelInvocationContext,
    ModelMessage,
    ModelPolicy,
    ModelRequest,
    ModelRole,
)
from app.core.models.gateway import ModelGateway
from app.core.models.providers.mock import MockProviderAdapter


@pytest.fixture
def sample_context():
    return ModelInvocationContext(
        tenant_id="school_1",
        execution_id="exec_stream_1",
        task_id="task_stream_1",
        correlation_id="corr_stream_1",
    )


@pytest.mark.asyncio
async def test_gateway_streaming_yields_chunks(sample_context):
    mock_provider = MockProviderAdapter(name="mock", default_response="Hello stream world")
    gateway = ModelGateway(providers=[mock_provider])

    request = ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="Stream test")],
        context=sample_context,
        policy=ModelPolicy(preferred_provider="mock"),
        stream=True,
    )

    chunks = []
    async for chunk in gateway.generate_stream(request):
        chunks.append(chunk.delta_content)

    full_text = "".join(chunks)
    assert full_text == "Hello stream world"


@pytest.mark.asyncio
async def test_gateway_streaming_cancellation_safety(sample_context):
    mock_provider = MockProviderAdapter(
        name="mock",
        default_response="A very long generated stream with many words and continuous output",
        latency_seconds=0.01,
    )
    gateway = ModelGateway(providers=[mock_provider])

    request = ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="Stream test")],
        context=sample_context,
        policy=ModelPolicy(preferred_provider="mock"),
        stream=True,
    )

    collected_chunks = []

    async def consume_stream():
        async for chunk in gateway.generate_stream(request):
            collected_chunks.append(chunk.delta_content)
            if len(collected_chunks) >= 2:
                raise asyncio.CancelledError("Client cancelled stream early")

    with pytest.raises(asyncio.CancelledError):
        await consume_stream()

    assert len(collected_chunks) >= 2
