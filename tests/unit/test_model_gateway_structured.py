"""AI-01 PR 2, decision 7: structured output is the gateway's.

A structured request asks for one JSON object. Whichever adapter answers, the
gateway parses and checks it, so no caller trusts a provider's JSON on its
own, and an answer that isn't one object fails the same way everywhere.
"""

import pytest

from app.core.models.budget import BudgetLimits, BudgetTracker
from app.core.models.contracts import (
    ModelInvocationContext,
    ModelMessage,
    ModelPolicy,
    ModelRequest,
    ModelRole,
    StructuredOutput,
)
from app.core.models.errors import (
    FallbackExhaustedError,
    InvalidRequestError,
    InvalidStructuredOutputError,
)
from app.core.models.gateway import ModelGateway
from app.core.models.providers.mock import MockProviderAdapter

MARKER = "passage-text-4b1d"
CONTEXT = ModelInvocationContext(
    tenant_id="school_1", execution_id="exec_1", task_id="t", correlation_id="c"
)


def request(*providers: str, structured: StructuredOutput | None = None) -> ModelRequest:
    return ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="Write questions")],
        context=CONTEXT,
        policy=ModelPolicy(preferred_provider=providers[0], fallback_providers=list(providers[1:])),
        structured=structured,
    )


QUESTIONS = StructuredOutput(kind="questions", hints={"count": 2})


def gateway(*adapters: MockProviderAdapter, tracker: BudgetTracker | None = None) -> ModelGateway:
    return ModelGateway(
        providers=list(adapters),
        approved_providers={a.name for a in adapters},
        budget_tracker=tracker,
    )


@pytest.mark.asyncio
async def test_a_structured_answer_comes_back_parsed():
    adapter = MockProviderAdapter(name="a", default_response='{"questions": [1, 2]}')

    response = await gateway(adapter).generate(request("a", structured=QUESTIONS))

    assert response.structured == {"questions": [1, 2]}
    assert response.content == '{"questions": [1, 2]}'


@pytest.mark.asyncio
async def test_an_unstructured_answer_is_not_parsed():
    adapter = MockProviderAdapter(name="a", default_response="plain words, not JSON")

    response = await gateway(adapter).generate(request("a"))

    assert response.structured is None
    assert response.content == "plain words, not JSON"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        f"Sure! Here are your questions: {MARKER}",  # not JSON
        f'["{MARKER}"]',  # JSON, but not an object
        '"' + MARKER + '"',
        "[" * 100_000 + "]" * 100_000,  # too deep to parse
        "",
    ],
)
async def test_an_answer_that_is_not_one_json_object_fails_without_quoting_it(answer):
    adapter = MockProviderAdapter(name="a", default_response=answer)
    tracker = BudgetTracker()
    tracker.set_budget("exec_1", BudgetLimits(max_model_calls=5))

    with pytest.raises(InvalidStructuredOutputError) as caught:
        await gateway(adapter, tracker=tracker).generate(request("a", structured=QUESTIONS))

    error = caught.value
    assert MARKER not in str(error) and MARKER not in repr(error.args)
    assert error.provider == "a"
    assert error.__cause__ is None and error.__context__ is None
    # Not retried (one call), and charged: the provider did answer.
    assert len(adapter.call_history) == 1
    assert tracker.get_usage("exec_1").model_calls == 1
    assert tracker.get_reserved("exec_1").model_calls == 0


@pytest.mark.asyncio
async def test_a_named_fallback_answers_when_the_preferred_answer_is_unusable():
    broken = MockProviderAdapter(name="a", default_response="not JSON")
    good = MockProviderAdapter(name="b", default_response='{"ok": true}')

    response = await gateway(broken, good).generate(request("a", "b", structured=QUESTIONS))

    assert response.provider == "b"
    assert response.structured == {"ok": True}


@pytest.mark.asyncio
async def test_unusable_answers_from_every_provider_exhaust_the_fallbacks():
    a = MockProviderAdapter(name="a", default_response="no")
    b = MockProviderAdapter(name="b", default_response="[]")

    with pytest.raises(FallbackExhaustedError) as caught:
        await gateway(a, b).generate(request("a", "b", structured=QUESTIONS))

    assert "InvalidStructuredOutputError" in str(caught.value)


@pytest.mark.asyncio
async def test_a_structured_request_is_refused_for_streaming_before_any_call():
    adapter = MockProviderAdapter(name="a", default_response='{"ok": true}')
    tracker = BudgetTracker()
    tracker.set_budget("exec_1", BudgetLimits(max_model_calls=5))

    with pytest.raises(InvalidRequestError):
        async for _ in gateway(adapter, tracker=tracker).generate_stream(
            request("a", structured=QUESTIONS)
        ):
            pass

    assert adapter.call_history == []
    assert tracker.get_reserved("exec_1").model_calls == 0
    assert tracker.get_usage("exec_1").model_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        '{"score": NaN}',
        '{"score": Infinity}',
        '{"score": -Infinity}',
        '{"scores": [1, {"deep": NaN}]}',
        '{"score": 1e400}',  # overflows to infinity
        '{"score": -1e400}',
    ],
)
async def test_values_python_accepts_but_json_does_not_are_refused(answer):
    adapter = MockProviderAdapter(name="a", default_response=answer)

    with pytest.raises(InvalidStructuredOutputError):
        await gateway(adapter).generate(request("a", structured=QUESTIONS))


@pytest.mark.asyncio
async def test_ordinary_numbers_are_kept_as_json_gives_them():
    answer = (
        '{"int": 10, "big": 123456789012345678901234567890, "float": 1.5, "exp": 2e3, "neg": -0.25}'
    )
    adapter = MockProviderAdapter(name="a", default_response=answer)

    response = await gateway(adapter).generate(request("a", structured=QUESTIONS))

    assert response.structured == {
        "int": 10,
        "big": 123456789012345678901234567890,
        "float": 1.5,
        "exp": 2000.0,
        "neg": -0.25,
    }
