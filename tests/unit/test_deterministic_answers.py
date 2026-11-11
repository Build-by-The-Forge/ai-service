"""The offline model's answers, pinned.

Deterministic mode answers chat, explain and question generation without a
model, and CI and the two-service boundary suite depend on exactly what it
says (a cited passage, questions in passage [1]'s words). These are its
answers as they were before it moved behind the gateway (AI-01 decision 6):
DeterministicLLM and the gateway's adapter both give exactly these.
"""

import pytest

import app.core.models.providers.deterministic as deterministic_module
from app.core.models.contracts import (
    ModelInvocationContext,
    ModelMessage,
    ModelPolicy,
    ModelRequest,
    ModelRole,
    StructuredOutput,
)
from app.core.models.gateway import ModelGateway
from app.core.models.providers.deterministic import DeterministicProviderAdapter
from app.services.ai.deterministic import DeterministicLLM

PASSAGES = (
    "Explain osmosis.\n\nCourse passages:\n"
    "[1] (Biology 101, p. 3) Osmosis moves water across a semipermeable membrane"
    " toward higher solute concentration\n\n"
    "[2] (Biology 101, p. 4) Diffusion spreads particles evenly."
)
NO_PASSAGES = "Explain osmosis."


def mcq(prompt: str, answer: str, explanation: str, difficulty: str, sources: list[int]):
    return {
        "question_type": "mcq",
        "prompt": prompt,
        "options": [{"text": answer, "is_correct": True}]
        + [{"text": f"Option {c}", "is_correct": False} for c in "BCD"],
        "explanation": explanation,
        "difficulty": difficulty,
        "sources": sources,
    }


TEXT_ANSWERS = [
    (PASSAGES, "[deterministic model] prompt sha256:48dab5b4d97403c7 (206 chars) [1]"),
    (NO_PASSAGES, "[deterministic model] prompt sha256:5934cef3b3572240 (16 chars)"),
]

STRUCTURED_ANSWERS = [
    (
        PASSAGES,
        "questions",
        {"count": 2, "difficulty": "hard"},
        {
            "questions": [
                mcq(
                    "[deterministic question] Osmosis moves water across"
                    " semipermeable membrane toward higher",
                    "solute concentration",
                    "[deterministic 48dab5b4] Explanation 1",
                    "hard",
                    [1],
                ),
                mcq(
                    "[deterministic question] moves water across semipermeable"
                    " membrane toward higher solute",
                    "solute concentration",
                    "[deterministic 48dab5b4] Explanation 2",
                    "hard",
                    [1],
                ),
            ]
        },
    ),
    (
        PASSAGES,
        "questions",
        {"count": 1, "question_type": "true_false"},
        {
            "questions": [
                {
                    "question_type": "true_false",
                    "prompt": "[deterministic question] Osmosis moves water across"
                    " semipermeable membrane toward higher",
                    "options": [
                        {"text": "True", "is_correct": True},
                        {"text": "False", "is_correct": False},
                    ],
                    "explanation": "[deterministic 48dab5b4] Explanation 1",
                    "difficulty": "medium",
                    "sources": [1],
                }
            ]
        },
    ),
    (
        "No passages.",
        "questions",
        {"count": 1},
        {
            "questions": [
                mcq(
                    "[deterministic question] Question 1",
                    "Option A",
                    "[deterministic a565551e] Explanation 1",
                    "medium",
                    [],
                )
            ]
        },
    ),
    (
        PASSAGES,
        "summary",
        {},
        {"answer": "[deterministic model] prompt sha256:48dab5b4d97403c7 (206 chars) [1]"},
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("prompt", "expected"), TEXT_ANSWERS)
async def test_the_text_answer(prompt, expected):
    assert await DeterministicLLM().generate(prompt) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("prompt", "kind", "hints", "expected"), STRUCTURED_ANSWERS)
async def test_the_structured_answer(prompt, kind, hints, expected):
    assert await DeterministicLLM().generate_json(prompt, kind=kind, **hints) == expected


# --- The same answers from the gateway's deterministic adapter (AI-01 decision 6)


def through_gateway(prompt: str, structured: StructuredOutput | None = None) -> ModelRequest:
    return ModelRequest(
        # A system message is the caller's framing: the prompt is the user's.
        messages=[
            ModelMessage(role=ModelRole.SYSTEM, content="You are an expert AI tutor."),
            ModelMessage(role=ModelRole.USER, content=prompt),
        ],
        context=ModelInvocationContext(
            tenant_id="school_1", execution_id="e", task_id="t", correlation_id="c"
        ),
        policy=ModelPolicy(preferred_provider="deterministic"),
        structured=structured,
    )


def gateway() -> ModelGateway:
    return ModelGateway(
        providers=[DeterministicProviderAdapter()], approved_providers={"deterministic"}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("prompt", "expected"), TEXT_ANSWERS)
async def test_the_adapter_gives_the_same_text_answer(prompt, expected):
    response = await gateway().generate(through_gateway(prompt))

    assert response.content == expected
    assert response.provider == "deterministic" and response.structured is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("prompt", "expected"), TEXT_ANSWERS)
async def test_the_adapter_streams_the_same_text_answer(prompt, expected):
    chunks = [c.delta_content async for c in gateway().generate_stream(through_gateway(prompt))]

    assert "".join(chunks) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("prompt", "kind", "hints", "expected"), STRUCTURED_ANSWERS)
async def test_the_adapter_gives_the_same_structured_answer(prompt, kind, hints, expected):
    response = await gateway().generate(
        through_gateway(prompt, StructuredOutput(kind=kind, hints=hints))
    )

    assert response.structured == expected


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_the_adapter_cannot_be_built_in_a_strict_environment(monkeypatch, environment):
    monkeypatch.setattr(deterministic_module.settings, "APP_ENVIRONMENT", environment)

    with pytest.raises(RuntimeError, match="development and tests only"):
        DeterministicProviderAdapter()
