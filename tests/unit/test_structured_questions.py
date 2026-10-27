"""Structured question generation (M5 design §3, M5-2)."""

import copy

import pytest

from app.services.ai.deterministic import DeterministicLLM
from app.services.ai.llm import LLMProvider
from app.services.generation.structured_questions import StructuredQuestionGenerator
from app.services.rag.evidence import Evidence, Passage


class NoRetrieval:
    """One course passage on the topic."""

    async def evidence(self, query):
        passage = Passage(1, f"context for {query}", "chunk-1", "asset-1", 1, 1)
        return Evidence(query=query, passages=(passage,))


GOOD = {
    "question_type": "mcq",
    "prompt": "Which organelle makes ATP?",
    "options": [
        {"text": "Nucleus", "is_correct": False},
        {"text": "Mitochondrion", "is_correct": True},
        {"text": "Ribosome", "is_correct": False},
    ],
    "explanation": "Mitochondria carry out respiration.",
    "difficulty": "medium",
}


def bad(**changes):
    item = copy.deepcopy(GOOD)
    item.update(changes)
    return item


class Scripted(LLMProvider):
    def __init__(self, answer):
        self.answer = answer
        self.prompts: list[str] = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return self.answer if isinstance(self.answer, str) else ""

    async def generate_json(self, prompt, *, kind, **hints):
        self.prompts.append(prompt)
        if isinstance(self.answer, str):
            return await super().generate_json(prompt, kind=kind, **hints)
        return self.answer


async def generate(llm, count=5, difficulty="medium", question_type="mcq"):
    return await StructuredQuestionGenerator(llm, NoRetrieval()).generate(
        "cells", count=count, difficulty=difficulty, question_type=question_type
    )


async def test_valid_questions_are_returned_and_the_prompt_carries_the_context():
    llm = Scripted({"questions": [GOOD, GOOD | {"prompt": "Second?"}]})
    result = await generate(llm)
    assert [q.prompt for q in result.questions] == ["Which organelle makes ATP?", "Second?"]
    assert result.discarded == 0
    assert "context for cells" in llm.prompts[0]


@pytest.mark.parametrize(
    "item",
    [
        bad(options=[{"text": "A", "is_correct": True}, {"text": "B", "is_correct": True}]),
        bad(options=[{"text": "A", "is_correct": False}, {"text": "B", "is_correct": False}]),
        bad(options=[{"text": "A", "is_correct": True}]),
        bad(options=[{"text": "A", "is_correct": True}, {"text": "a", "is_correct": False}]),
        bad(prompt=""),
        bad(question_type="essay"),
        bad(difficulty="hard"),
        bad(tenant_id="other-tenant"),
        {"prompt": "no options"},
        "a string",
    ],
    ids=[
        "two-correct",
        "none-correct",
        "one-option",
        "duplicate-options",
        "empty-prompt",
        "unknown-type",
        "unrequested-difficulty",
        "extra-field",
        "missing-fields",
        "not-an-object",
    ],
)
async def test_invalid_items_are_discarded_never_repaired(item):
    result = await generate(Scripted({"questions": [GOOD, item]}))
    assert len(result.questions) == 1
    assert result.discarded == 1


async def test_true_false_needs_exactly_true_and_false():
    tf = bad(
        question_type="true_false",
        options=[{"text": "True", "is_correct": True}, {"text": "False", "is_correct": False}],
    )
    wrong = bad(
        question_type="true_false",
        options=[{"text": "Yes", "is_correct": True}, {"text": "No", "is_correct": False}],
    )
    result = await generate(Scripted({"questions": [tf, wrong]}), question_type="true_false")
    assert len(result.questions) == 1 and result.discarded == 1


async def test_extra_questions_beyond_the_count_are_dropped():
    result = await generate(Scripted({"questions": [GOOD] * 4}), count=2)
    assert len(result.questions) == 2 and result.discarded == 2


@pytest.mark.parametrize("answer", ["not json", "[1, 2]", {"items": []}])
async def test_an_unusable_answer_yields_nothing(answer):
    result = await generate(Scripted(answer), count=3)
    assert result.questions == [] and result.discarded == 3


@pytest.mark.parametrize("question_type", ["mcq", "true_false"])
async def test_the_deterministic_provider_produces_valid_questions(question_type):
    result = await generate(
        DeterministicLLM(), count=4, difficulty="hard", question_type=question_type
    )
    assert len(result.questions) == 4 and result.discarded == 0
    assert all(
        q.difficulty == "hard" and q.question_type == question_type for q in result.questions
    )
