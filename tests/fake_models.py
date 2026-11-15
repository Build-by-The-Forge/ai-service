"""Test-only stand-ins for the model interface the orchestrator and the
question generators take (the one GatewayLLM provides in the app).

In app/, every model call goes through GatewayLLM and the gateway. These
fakes let unit tests script or record answers without it:

- FakeModel: subclass it and define generate(). for_capability() yields the
  fake itself, and generate_json() parses generate()'s text as one JSON
  object, as the old LLMProvider defaults did.
- OfflineModel: the gateway's deterministic adapter's answers, given
  directly, as DeterministicLLM gave them. It is for tests that compare the
  gateway path with a direct answer, or that need offline answers without
  a gateway.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.core.models.contracts import ModelInvocationContext
from app.core.models.providers.deterministic import questions_answer, text_answer
from app.services.ai.llm import StructuredOutputError


class FakeModel:
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError

    @contextmanager
    def for_capability(
        self, capability: str, invocation: ModelInvocationContext | None
    ) -> Iterator["FakeModel"]:
        yield self

    async def generate_json(self, prompt: str, *, kind: str, **hints: Any) -> dict[str, Any]:
        text = await self.generate(prompt)
        try:
            value = json.loads(text)
        except ValueError as exc:
            raise StructuredOutputError("model answer is not JSON") from exc
        if not isinstance(value, dict):
            raise StructuredOutputError("model answer is not a JSON object")
        return value


class OfflineModel(FakeModel):
    async def generate(self, prompt: str) -> str:
        return text_answer(prompt)

    async def generate_json(self, prompt: str, *, kind: str, **hints: Any) -> dict[str, Any]:
        if kind == "questions":
            return questions_answer(prompt, hints)
        return {"answer": await self.generate(prompt)}
