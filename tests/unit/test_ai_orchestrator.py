"""Chat, explanations and question generation answer from the tenant's
course passages, citing them, or not at all (P3-5 grounding)."""

import pytest

from app.services.ai.llm import LLMProvider
from app.services.ai.orchestrator import AIOrchestrator
from app.services.rag.evidence import NOT_COVERED, Evidence, Passage

OSMOSIS = Passage(
    number=1,
    text="Osmosis is the diffusion of water molecules across a partially permeable membrane.",
    chunk_id="chunk-1",
    document_id="asset-bio101",
    page_start=2,
    page_end=2,
)


class ScriptedLLM(LLMProvider):
    def __init__(self, reply: str = "Water moves across the membrane [1]."):
        self.reply = reply
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


class EvidenceRetriever:
    def __init__(self, *passages: Passage):
        self.passages = passages
        self.queries: list[str] = []

    async def evidence(self, query: str) -> Evidence:
        self.queries.append(query)
        return Evidence(query=query, passages=self.passages)


class DummyMemory:
    async def get_user_memory(self, user_id: str):
        return [{"topic": "math", "mastery_score": 0.8}]


def orchestrator(llm=None, *passages):
    return AIOrchestrator(
        llm=llm or ScriptedLLM(), retriever=EvidenceRetriever(*passages), memory=DummyMemory()
    )


async def test_a_cited_answer_is_kept_with_its_citations_and_full_passages():
    llm = ScriptedLLM()

    result = await orchestrator(llm, OSMOSIS).chat("user-1", "What is osmosis?")

    assert result["answer"] == "Water moves across the membrane [1]."
    assert result["grounded"] is True
    assert result["citations"] == [
        {
            "number": 1,
            "chunk_id": "chunk-1",
            "document_id": "asset-bio101",
            "page_start": 2,
            "page_end": 2,
        }
    ]
    # The model saw the whole passage, numbered, with its document and page.
    assert f"[1] (asset-bio101, p. 2) {OSMOSIS.text}" in llm.prompts[0]
    assert result["context_used"][0]["text"] == OSMOSIS.text
    assert "math" in result["memory_used"]


@pytest.mark.parametrize(
    "reply",
    ["Osmosis is when cells drink.", "Water moves [2].", "See [1] and [3]."],
    ids=["no-citation", "cites-a-passage-it-was-not-given", "one-unknown-citation"],
)
async def test_an_answer_not_grounded_in_the_passages_is_replaced(reply):
    result = await orchestrator(ScriptedLLM(reply), OSMOSIS).chat("user-1", "What is osmosis?")

    assert (result["answer"], result["grounded"], result["citations"]) == (NOT_COVERED, False, [])


async def test_without_evidence_the_model_is_not_asked():
    llm = ScriptedLLM()

    result = await orchestrator(llm).chat("user-1", "Who won the World Cup?")

    assert (result["answer"], result["grounded"], result["context_used"]) == (
        NOT_COVERED,
        False,
        [],
    )
    assert llm.prompts == []


async def test_explanations_are_grounded_the_same_way():
    grounded = await orchestrator(ScriptedLLM(), OSMOSIS).explain("osmosis")
    uncovered = await orchestrator(ScriptedLLM()).explain("quantum chromodynamics")

    assert (grounded["grounded"], grounded["citations"][0]["page_start"]) == (True, 2)
    assert (uncovered["explanation"], uncovered["grounded"]) == (NOT_COVERED, False)


async def test_no_questions_are_generated_without_course_material():
    llm = ScriptedLLM()

    result = await orchestrator(llm).generate_questions("geometry")

    assert result == {"questions": [], "discarded": 0, "grounded": False, "rejected": {}}
    assert llm.prompts == []


async def test_questions_are_generated_from_the_passages():
    llm = ScriptedLLM()

    result = await orchestrator(llm, OSMOSIS).generate_questions("osmosis")

    # The scripted model does not answer with JSON: all discarded, but grounded.
    assert result == {"questions": [], "discarded": 5, "grounded": True, "rejected": {}}
    assert OSMOSIS.text in llm.prompts[0]
