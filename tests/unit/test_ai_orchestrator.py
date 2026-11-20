"""Chat, explanations and question generation answer from the tenant's
course passages, citing them, or not at all (P3-5 grounding)."""

import pytest

from tests.fake_models import FakeModel
from app.services.ai.orchestrator import AIOrchestrator
from app.services.rag.evidence import NOT_COVERED, Evidence, Passage
from tests.rag_support import sources_for

OSMOSIS = Passage(
    number=1,
    text="Osmosis is the diffusion of water molecules across a partially permeable membrane.",
    chunk_id="chunk-1",
    document_id="asset-bio101",
    page_start=2,
    page_end=2,
)


class ScriptedLLM(FakeModel):
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

    async def evidence(self, query: str, denied_assets: frozenset[str]) -> Evidence:
        self.queries.append(query)
        return Evidence(
            query=query, passages=self.passages, sources_used=sources_for(*self.passages)
        )


class DummyMemory:
    async def get_user_memory(self, user_id: str):
        return [{"topic": "math", "mastery_score": 0.8}]


def orchestrator(llm=None, *passages):
    return AIOrchestrator(
        llm=llm or ScriptedLLM(), retriever=EvidenceRetriever(*passages), memory=DummyMemory()
    )


async def test_a_cited_answer_is_kept_with_its_citations_and_full_passages():
    llm = ScriptedLLM()

    result = await orchestrator(llm, OSMOSIS).chat("user-1", "What is osmosis?", frozenset())

    assert result["answer"] == "Water moves across the membrane [1]."
    assert result["grounded"] is True
    assert result["citations"] == [
        {
            "number": 1,
            "chunk_id": "chunk-1",
            "document_id": "asset-bio101",
            "page_start": 2,
            "page_end": 2,
            # M3-F: what the citation quotes, and the version it came from.
            "excerpt": OSMOSIS.text,
            "run_number": 1,
            "attempt": 1,
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
    result = await orchestrator(ScriptedLLM(reply), OSMOSIS).chat(
        "user-1", "What is osmosis?", frozenset()
    )

    assert (result["answer"], result["grounded"], result["citations"]) == (NOT_COVERED, False, [])


async def test_without_evidence_the_model_is_not_asked():
    llm = ScriptedLLM()

    result = await orchestrator(llm).chat("user-1", "Who won the World Cup?", frozenset())

    assert (result["answer"], result["grounded"], result["context_used"]) == (
        NOT_COVERED,
        False,
        [],
    )
    assert llm.prompts == []


async def test_explanations_are_grounded_the_same_way():
    grounded = await orchestrator(ScriptedLLM(), OSMOSIS).explain("osmosis", frozenset())
    uncovered = await orchestrator(ScriptedLLM()).explain("quantum chromodynamics", frozenset())

    assert (grounded["grounded"], grounded["citations"][0]["page_start"]) == (True, 2)
    assert (uncovered["explanation"], uncovered["grounded"]) == (NOT_COVERED, False)


async def test_no_questions_are_generated_without_course_material():
    llm = ScriptedLLM()

    result = await orchestrator(llm).generate_questions("geometry", frozenset())

    assert result == {
        "questions": [],
        "discarded": 0,
        "grounded": False,
        "rejected": {},
        "sources_used": [],
    }
    assert llm.prompts == []


async def test_questions_are_generated_from_the_passages():
    llm = ScriptedLLM()

    result = await orchestrator(llm, OSMOSIS).generate_questions("osmosis", frozenset())

    # The scripted model does not answer with JSON: all discarded, but grounded.
    assert result == {
        "questions": [],
        "discarded": 5,
        "grounded": True,
        "rejected": {},
        # Retrieval supplied the passage, so it is a source whatever the model did.
        "sources_used": [{"asset_id": "asset-bio101", "run_number": 1, "attempt": 1}],
    }
    assert OSMOSIS.text in llm.prompts[0]


# --- M3-F: one tutor turn ---------------------------------------------------------------

PLANTS = Passage(
    number=2,
    text="Osmotic pressure keeps plant cells turgid against the cell wall.",
    chunk_id="chunk-2",
    document_id="asset-bot102",
    page_start=None,
    page_end=None,
)


class NoMemoryReads:
    async def get_user_memory(self, user_id: str):
        raise AssertionError("a tutor turn reads no learning memory")


def tutor(llm, *passages):
    retriever = EvidenceRetriever(*passages)
    return AIOrchestrator(llm=llm, retriever=retriever, memory=NoMemoryReads()), retriever


async def test_a_turn_returns_its_cited_passages_and_every_source_but_no_context():
    llm = ScriptedLLM("Water moves across the membrane [1].")
    turn, _ = tutor(llm, OSMOSIS, PLANTS)

    result = await turn.tutor_turn("What is osmosis?", [], frozenset())

    assert set(result) == {"answer", "grounded", "citations", "sources_used"}
    assert result["grounded"] is True
    assert [c["number"] for c in result["citations"]] == [1]
    assert result["citations"][0]["excerpt"] == OSMOSIS.text
    # The uncited passage is a source, never returned as text.
    assert {s["asset_id"] for s in result["sources_used"]} == {"asset-bio101", "asset-bot102"}
    assert PLANTS.text not in repr(result)


async def test_a_turn_retrieves_on_the_message_alone_and_prompts_with_the_earlier_questions():
    llm = ScriptedLLM()
    turn, retriever = tutor(llm, OSMOSIS)

    await turn.tutor_turn("And in plants?", ["What is osmosis?", "Is it passive?"], frozenset())

    assert retriever.queries == ["And in plants?"]
    [prompt] = llm.prompts
    assert "- What is osmosis?" in prompt and "- Is it passive?" in prompt
    assert prompt.index("Is it passive?") < prompt.index("COURSE PASSAGES")
    assert "STUDENT MEMORY" not in prompt


async def test_a_turn_prompt_names_no_document_so_the_answer_cannot_copy_one():
    llm = ScriptedLLM()
    turn, _ = tutor(llm, OSMOSIS, PLANTS)

    await turn.tutor_turn("What is osmosis?", [], frozenset())

    [prompt] = llm.prompts
    assert f"[1] (p. 2) {OSMOSIS.text}" in prompt
    assert f"[2] (page unknown) {PLANTS.text}" in prompt
    for internal in ("asset-bio101", "asset-bot102", "chunk-1", "chunk-2"):
        assert internal not in prompt


@pytest.mark.parametrize(
    "reply",
    ["Osmosis is when cells drink.", "Water moves [3].", "See [1] and [3]."],
    ids=["no-citation", "cites-a-passage-it-was-not-given", "one-unknown-citation"],
)
async def test_an_ungrounded_turn_is_not_covered_with_no_citations(reply):
    turn, _ = tutor(ScriptedLLM(reply), OSMOSIS, PLANTS)

    result = await turn.tutor_turn("What is osmosis?", [], frozenset())

    assert (result["answer"], result["grounded"], result["citations"]) == (NOT_COVERED, False, [])
    # Retrieval still supplied them: the sources stand whatever the model did.
    assert len(result["sources_used"]) == 2


async def test_without_evidence_a_turn_does_not_ask_the_model():
    llm = ScriptedLLM()
    turn, _ = tutor(llm)

    result = await turn.tutor_turn("Who won the World Cup?", ["Hello?"], frozenset())

    assert result == {
        "answer": NOT_COVERED,
        "grounded": False,
        "citations": [],
        "sources_used": [],
    }
    assert llm.prompts == []
