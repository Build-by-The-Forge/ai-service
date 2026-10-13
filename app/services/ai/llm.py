import json
from abc import ABC, abstractmethod
from typing import Any


class StructuredOutputError(ValueError):
    """The model's answer was not the JSON object that was asked for."""


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass

    async def generate_json(self, prompt: str, *, kind: str, **hints: Any) -> dict[str, Any]:
        """A JSON object answer. ``kind`` and ``hints`` (such as count) let
        offline providers answer without a model; real providers rely on the
        prompt alone. Raises StructuredOutputError for anything else."""
        text = await self.generate(prompt)
        try:
            value = json.loads(text)
        except ValueError as exc:
            raise StructuredOutputError("model answer is not JSON") from exc
        if not isinstance(value, dict):
            raise StructuredOutputError("model answer is not a JSON object")
        return value
