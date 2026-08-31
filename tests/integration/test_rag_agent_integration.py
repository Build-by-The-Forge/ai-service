import json

import pytest

from app.core.agents.contracts import AgentContext, AgentExecutionPolicy
from app.core.agents.registry import AgentRegistry
from app.core.models.contracts import ModelDecision, ModelRequest, ModelResponse
from app.core.models.gateway import ModelGateway
from app.core.models.provider import ModelProviderAdapter
from app.core.rag.contracts import ChunkProvenance, ContentAuthority
from app.core.tools.registry import ToolRegistry
from app.services.agents.executor import AgentExecutor
from app.services.rag.retrieval import HybridRetrievalEngine, set_retrieval_engine
from app.services.tools.builtins.content import (
    CONTENT_SEARCH_DEFINITION,
    content_search_handler,
)
from app.services.tools.executor import ToolExecutor


class RAGScriptedModelProvider(ModelProviderAdapter):
    """
    Scripted provider that calls content.search and synthesizes a grounded answer with citations.
    """

    def __init__(self):
        self.call_count = 0

    @property
    def name(self) -> str:
        return "rag_scripted"

    def supports_model(self, model: str) -> bool:
        return True

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        self.call_count += 1
        # Step 1: Agent decides to search content
        if self.call_count == 1:
            tool_call = {
                "id": "call_rag_001",
                "type": "function",
                "function": {
                    "name": "content.search",
                    "arguments": json.dumps(
                        {"query": "What is Amdahl's Law?", "course_code": "CSC305"}
                    ),
                },
            }
            return ModelResponse(
                content=None,
                tool_calls=[tool_call],
                model="gpt-4o-mini",
                provider="rag_scripted",
            )
        # Step 2: Agent synthesizes grounded answer
        return ModelResponse(
            content="Based on the Computer Architecture Course Pack, Amdahl's Law predicts the theoretical speedup in latency of the execution of a task at fixed workload that can be expected of a system whose resources are improved.",
            tool_calls=None,
            model="gpt-4o-mini",
            provider="rag_scripted",
        )

    async def generate_stream(self, request: ModelRequest, decision: ModelDecision):
        yield None


@pytest.mark.asyncio
async def test_agent_to_tool_to_rag_hybrid_retrieval():
    # 1. Setup RAG Engine and index knowledge
    engine = HybridRetrievalEngine()
    set_retrieval_engine(engine)

    prov = ChunkProvenance(
        source_uri="s3://build-assets/csc305_arch.pdf",
        page_number=45,
        section_title="Parallel Performance Laws",
        authority=ContentAuthority.APPROVED_REFERENCE,
    )
    await engine.index_document(
        tenant_id="school_unilag",
        document_id="doc_csc305",
        document_title="Computer Architecture Course Pack",
        text="Amdahl's Law states that the overall performance improvement gained by optimizing a single part of a system is limited by the fraction of time that the improved part is actually used.",
        course_code="CSC305",
        topic="Parallel Computing",
        provenance=prov,
    )

    # 2. Setup Tool Registry & Executor
    tool_reg = ToolRegistry()
    tool_reg.register(CONTENT_SEARCH_DEFINITION, content_search_handler)
    tool_executor = ToolExecutor(registry=tool_reg)

    # 3. Setup Model Gateway & Agent Registry
    provider = RAGScriptedModelProvider()
    gateway = ModelGateway(providers=[provider])
    agent_reg = AgentRegistry()

    executor = AgentExecutor(
        agent_registry=agent_reg,
        tool_registry=tool_reg,
        model_gateway=gateway,
        tool_executor=tool_executor,
    )

    context = AgentContext(
        tenant_id="school_unilag",
        execution_id="exec_rag_100",
        task_id="task_rag_100",
        correlation_id="corr_rag_100",
        actor_role="student",
        permissions=["curriculum:read", "content:read", "assessment:generate"],
        capabilities=["curriculum_intelligence", "content_search", "cbt_generation"],
    )

    result = await executor.run(
        agent_id="study.assistant",
        prompt="Explain Amdahl's law for my CSC305 exam.",
        context=context,
        custom_policy=AgentExecutionPolicy(max_steps=5),
    )

    assert result.state.value == "COMPLETED"
    assert "Amdahl's Law" in (result.output or "")
    assert result.tool_call_count == 1
    assert result.step_count >= 1


@pytest.mark.asyncio
async def test_agent_tool_rag_cross_tenant_isolation():
    engine = HybridRetrievalEngine()
    set_retrieval_engine(engine)

    # Document belongs to school_futa
    await engine.index_document(
        tenant_id="school_futa",
        document_id="doc_secret_exam",
        document_title="FUTA Secret Exam Questions",
        text="Confidential Physics 101 question bank.",
        course_code="PHY101",
    )

    tool_reg = ToolRegistry()
    tool_reg.register(CONTENT_SEARCH_DEFINITION, content_search_handler)
    tool_executor = ToolExecutor(registry=tool_reg)

    # ToolContext is for school_unilag
    from app.core.tools.contracts import ToolContext, ToolInvocation

    tool_ctx = ToolContext(
        tenant_id="school_unilag",
        execution_id="exec_leak_test",
        task_id="task_leak_test",
        correlation_id="corr_leak_test",
        actor_role="student",
        permissions=["content:read"],
        capabilities=["content_search"],
    )

    inv = ToolInvocation(
        tool_name="content.search",
        arguments={"query": "Confidential Physics question bank"},
        context=tool_ctx,
    )

    result = await tool_executor.execute(inv)
    assert result.success is True
    assert result.data["results_count"] == 0
    assert len(result.data["results"]) == 0
