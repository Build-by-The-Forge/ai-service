"""OpenAILLM.generate_json calls the installed openai client correctly.

The call is bound against the real AsyncCompletions.create signature of the
pinned client, so an argument it does not accept fails here, not at runtime
(M5 passed response_format, which openai 1.0.0 does not have).
"""

import inspect
import json
import types

import pytest
from openai.resources.chat.completions import AsyncCompletions

from app.services.ai.llm import StructuredOutputError
from app.services.ai.openai import OpenAILLM


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content=self.content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def llm_with(content) -> tuple[OpenAILLM, FakeCompletions]:
    llm = OpenAILLM.__new__(OpenAILLM)
    completions = FakeCompletions(content)
    llm.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    return llm, completions


async def test_json_mode_is_requested_in_a_form_the_pinned_client_accepts():
    llm, completions = llm_with(json.dumps({"questions": []}))

    assert await llm.generate_json("prompt", kind="questions", count=2) == {"questions": []}

    [call] = completions.calls
    inspect.signature(AsyncCompletions.create).bind(object(), **call)  # raises if not accepted
    assert call["extra_body"] == {"response_format": {"type": "json_object"}}


@pytest.mark.parametrize("content", ["not json", "[1, 2]", None])
async def test_a_non_object_answer_is_a_structured_output_error(content):
    llm, _ = llm_with(content)
    with pytest.raises(StructuredOutputError):
        await llm.generate_json("prompt", kind="questions")


async def test_plain_generation_also_matches_the_client_signature():
    llm, completions = llm_with("an answer")
    assert await llm.generate("prompt") == "an answer"
    inspect.signature(AsyncCompletions.create).bind(object(), **completions.calls[0])
