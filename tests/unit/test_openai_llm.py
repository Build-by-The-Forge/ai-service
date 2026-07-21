import pytest

from app.services.ai import openai as openai_module
from app.services.ai.openai import OpenAILLM


class DummyEmptyResponse:
    def __init__(self):
        self.choices = [type("Msg", (), {"message": type("Content", (), {"content": None})()})()]


class DummyAsyncOpenAI:
    def __init__(self, api_key=None):
        class Completions:
            async def create(self, model, messages):
                return DummyEmptyResponse()

        self.chat = type("Chat", (), {"completions": Completions()})()


@pytest.mark.asyncio
async def test_openai_llm_raises_when_content_empty(monkeypatch):
    monkeypatch.setattr(openai_module, "AsyncOpenAI", DummyAsyncOpenAI)

    openai = OpenAILLM()

    with pytest.raises(ValueError, match="OpenAI response content was empty"):
        await openai.generate("prompt")
