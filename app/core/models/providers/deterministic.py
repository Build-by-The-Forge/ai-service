"""The offline model (MODEL_PROVIDER=deterministic): no network and no model.

A gateway provider adapter, so development, CI and the two-service boundary
suite exercise the real gateway path: eligibility, deadlines, budgets and
structured output. For development and tests only. It can't be built in
staging or production, and "deterministic" isn't a production provider
(GATEWAY_PROVIDERS), so a strict environment can't approve it either.

An answer names a digest and the size of the prompt, so a test can see which
prompt reached the model layer, without any real tutoring. Given numbered
course passages it behaves as a compliant model under the grounding rules:
it cites passage [1], and writes questions in passage [1]'s words.
"""

import hashlib
import json
import re
from collections.abc import AsyncIterator, Mapping
from typing import Any

from app.core.config import STRICT_ENVIRONMENTS, settings
from app.core.models.contracts import (
    ModelDecision,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelStreamChunk,
    ModelUsage,
)
from app.core.models.provider import ModelProviderAdapter

NAME = "deterministic"

_PASSAGE_ONE = re.compile(r"\n\[1\] \([^)]*\) (.*?)(?:\n\n\[2\] |\Z)", re.S)
_WORD = re.compile(r"[^\W\d_]{4,}")


def _first_passage_words(prompt: str) -> list[str]:
    """Words (no numbers) of the first numbered course passage in a prompt."""
    match = _PASSAGE_ONE.search(prompt)
    return _WORD.findall(match.group(1)) if match else []


def text_answer(prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    answer = f"[deterministic model] prompt sha256:{digest} ({len(prompt)} chars)"
    if "\n[1] (" in prompt:
        answer += " [1]"
    return answer


def questions_answer(prompt: str, hints: Mapping[str, Any]) -> dict[str, Any]:
    """Well-formed questions, labelled as deterministic so they are never
    mistaken for content."""
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
    count = int(hints.get("count", 1))
    true_false = hints.get("question_type") == "true_false"
    words = _first_passage_words(prompt)
    questions = []
    for i in range(1, count + 1):
        stem = " ".join(words[(i - 1) % max(1, len(words)) :][:8]) or f"Question {i}"
        answer = " ".join(words[8:12]) or "Option A"
        if true_false:
            options = [
                {"text": "True", "is_correct": i % 2 == 1},
                {"text": "False", "is_correct": i % 2 == 0},
            ]
        else:
            options = [{"text": answer, "is_correct": True}] + [
                {"text": f"Option {c}", "is_correct": False} for c in "BCD"
            ]
        questions.append(
            {
                "question_type": "true_false" if true_false else "mcq",
                "prompt": f"[deterministic question] {stem}",
                "options": options,
                "explanation": f"[deterministic {digest}] Explanation {i}",
                "difficulty": hints.get("difficulty", "medium"),
                "sources": [1] if words else [],
            }
        )
    return {"questions": questions}


def structured_answer(prompt: str, kind: str, hints: Mapping[str, Any]) -> dict[str, Any]:
    if kind == "questions":
        return questions_answer(prompt, hints)
    return {"answer": text_answer(prompt)}


def _prompt(request: ModelRequest) -> str:
    """The prompt is the last user message: system instructions are the
    gateway caller's framing, not what the model was asked."""
    for message in reversed(request.messages):
        if message.role == ModelRole.USER:
            return message.content
    return ""


def _tokens(text: str) -> int:
    return len(text) // 4 + 1


class DeterministicProviderAdapter(ModelProviderAdapter):
    def __init__(self) -> None:
        if settings.APP_ENVIRONMENT in STRICT_ENVIRONMENTS:
            raise RuntimeError("The deterministic model adapter is for development and tests only")

    @property
    def name(self) -> str:
        return NAME

    def supports_model(self, model: str) -> bool:
        return True

    def _content(self, request: ModelRequest) -> tuple[str, str]:
        prompt = _prompt(request)
        if request.structured is not None:
            value = structured_answer(prompt, request.structured.kind, request.structured.hints)
            return prompt, json.dumps(value)
        return prompt, text_answer(prompt)

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        prompt, content = self._content(request)
        prompt_tokens, completion_tokens = _tokens(prompt), _tokens(content)
        return ModelResponse(
            content=content,
            model=NAME,
            provider=NAME,
            role=ModelRole.ASSISTANT,
            finish_reason="stop",
            usage=ModelUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            latency_ms=0.0,
            context=request.context,
        )

    async def generate_stream(
        self, request: ModelRequest, decision: ModelDecision
    ) -> AsyncIterator[ModelStreamChunk]:
        _, content = self._content(request)
        words = content.split(" ")
        for i, word in enumerate(words):
            yield ModelStreamChunk(
                delta_content=word if i == 0 else f" {word}",
                model=NAME,
                provider=NAME,
                finish_reason="stop" if i == len(words) - 1 else None,
            )
