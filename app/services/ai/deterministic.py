from typing import Any

from app.core.models.providers.deterministic import questions_answer, text_answer
from app.services.ai.llm import LLMProvider


class DeterministicLLM(LLMProvider):
    """Offline stand-in for a model (MODEL_PROVIDER=deterministic), through
    the LLMProvider interface. Development and testing only.

    Its answers are the gateway's deterministic adapter's
    (app/core/models/providers/deterministic.py); this class keeps the old
    interface only until chat, explain and question generation move onto the
    gateway (AI-01 PR 3).
    """

    async def generate(self, prompt: str) -> str:
        return text_answer(prompt)

    async def generate_json(self, prompt: str, *, kind: str, **hints: Any) -> dict[str, Any]:
        if kind == "questions":
            return questions_answer(prompt, hints)
        return {"answer": await self.generate(prompt)}
