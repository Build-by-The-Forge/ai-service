import pytest

from app.services.ai.llm import LLMProvider
from app.services.ai.orchestrator import AIOrchestrator


class DummyLLM(LLMProvider):
    async def generate(self, prompt: str) -> str:
        return f"answer:{prompt}"


class DummyRetriever:
    async def retrieve(self, query: str):
        return [{"text": f"doc for {query}"}]


class DummyMemory:
    async def get_user_memory(self, user_id: str):
        return [{"topic": "math", "mastery_score": 0.8}]


@pytest.mark.asyncio
async def test_chat_returns_answer_and_context():
    orchestrator = AIOrchestrator(
        llm=DummyLLM(),
        retriever=DummyRetriever(),
        memory=DummyMemory(),
    )

    result = await orchestrator.chat("user-1", "What is 2+2?")

    assert result["answer"].startswith("answer:")
    assert result["context_used"][0]["text"].startswith("doc")
    assert "math" in result["memory_used"]


@pytest.mark.asyncio
async def test_explain_uses_llm_prompt():
    orchestrator = AIOrchestrator(llm=DummyLLM(), retriever=DummyRetriever(), memory=DummyMemory())

    result = await orchestrator.explain("algebra")

    assert "algebra" in result["explanation"]


@pytest.mark.asyncio
async def test_generate_questions_returns_question_payload():
    orchestrator = AIOrchestrator(llm=DummyLLM(), retriever=DummyRetriever(), memory=DummyMemory())

    result = await orchestrator.generate_questions("geometry")

    # A model that does not answer with JSON yields no questions, all discarded.
    assert result == {"questions": [], "discarded": 5}
