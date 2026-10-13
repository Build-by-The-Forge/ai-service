import json
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.services.ai.llm import LLMProvider, StructuredOutputError


class OpenAILLM(LLMProvider):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate(self, prompt: str) -> str:
        res = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert AI tutor."},
                {"role": "user", "content": prompt},
            ],
        )

        content = res.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI response content was empty")
        return content

    async def generate_json(self, prompt: str, *, kind: str, **hints: Any) -> dict[str, Any]:
        res = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            # JSON mode. The pinned openai client (1.0.0) predates the
            # response_format parameter, so it travels in the request body.
            extra_body={"response_format": {"type": "json_object"}},
            messages=[
                {"role": "system", "content": "You answer with one JSON object only."},
                {"role": "user", "content": prompt},
            ],
        )
        content = res.choices[0].message.content
        try:
            value = json.loads(content or "")
        except ValueError as exc:
            raise StructuredOutputError("model answer is not JSON") from exc
        if not isinstance(value, dict):
            raise StructuredOutputError("model answer is not a JSON object")
        return value
