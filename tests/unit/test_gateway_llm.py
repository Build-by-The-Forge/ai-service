"""AI-01 PR 3: GatewayLLM, the LLMProvider interface over the gateway.

Chat, explain and question generation call the model through it, so every
call is the gateway's: its approval check, deadline, retries, budgets,
structured output and sanitized errors (decisions P3-1 to P3-4).
"""

import logging
import time

import pytest

import app.services.ai.gateway_llm as gateway_llm
from app.core.models.budget import BudgetTracker
from app.core.models.contracts import (
    ModelDecision,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelUsage,
)
from app.core.models.errors import (
    BudgetExceededError,
    NoEligibleProviderError,
    ProviderUnavailableError,
)
from app.core.models.gateway import ModelGateway
from app.core.models.provider import ModelProviderAdapter
from app.core.models.providers.deterministic import DeterministicProviderAdapter
from app.core.service_signature_v1 import VerifiedCall
from app.services.ai.deterministic import DeterministicLLM
from app.services.ai.gateway_llm import GatewayLLM, capability_limits, invocation_context
from app.services.ai.llm import StructuredOutputError
from tests.signing import TENANT_A, user_context

PASSAGES = (
    "Explain osmosis.\n\nCourse passages:\n"
    "[1] (Biology 101, p. 3) Osmosis moves water across a semipermeable membrane"
    " toward higher solute concentration\n\n"
    "[2] (Biology 101, p. 4) Diffusion spreads particles evenly."
)
MARKER = "student-passage-61d4"


def call(request_id: str = "8f14e45f-ceea-4672-8a2b-0e3a3b4c5d6e") -> VerifiedCall:
    return VerifiedCall(
        sender="build-backend",
        timestamp=int(time.time()),
        request_id=request_id,
        context=user_context(tenant_id=TENANT_A, permissions=("ai.use",)),
        raw_context=b"",
    )


class Recording(ModelProviderAdapter):
    """Answers with fixed content, recording every request; can fail first."""

    def __init__(self, name: str = "rec", content: str = "Answer [1].", failures: int = 0):
        self._name, self.content, self.failures = name, content, failures
        self.requests: list[ModelRequest] = []

    @property
    def name(self) -> str:
        return self._name

    def supports_model(self, model: str) -> bool:
        return True

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        self.requests.append(request)
        if self.failures:
            self.failures -= 1
            raise ProviderUnavailableError(f"down: {MARKER}", provider=self._name)
        return ModelResponse(
            content=self.content,
            model=decision.model,
            provider=self._name,
            role=ModelRole.ASSISTANT,
            usage=ModelUsage(total_tokens=10),
            context=request.context,
        )

    async def generate_stream(self, request, decision):  # pragma: no cover - not used
        raise NotImplementedError
        yield


@pytest.fixture
def provider(monkeypatch):
    """The configured provider, as MODEL_PROVIDER names it."""

    def configure(name: str) -> None:
        monkeypatch.setattr(gateway_llm.settings, "MODEL_PROVIDER", name)

    return configure


def gateway(*adapters: ModelProviderAdapter, approved: set[str] | None = None) -> ModelGateway:
    return ModelGateway(
        providers=list(adapters),
        approved_providers={a.name for a in adapters} if approved is None else approved,
        budget_tracker=BudgetTracker(),
    )


# --- deterministic mode answers exactly as before ---------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", [PASSAGES, "Explain osmosis."])
async def test_deterministic_mode_answers_through_the_gateway_as_it_did_before(prompt, provider):
    provider("deterministic")
    llm = GatewayLLM(gateway(DeterministicProviderAdapter()))

    with llm.for_capability("chat", invocation_context(call(), "chat")) as bound:
        assert await bound.generate(prompt) == await DeterministicLLM().generate(prompt)


@pytest.mark.asyncio
async def test_deterministic_questions_come_through_the_gateway_as_they_did_before(provider):
    provider("deterministic")
    llm = GatewayLLM(gateway(DeterministicProviderAdapter()))
    hints = {"count": 2, "difficulty": "hard", "question_type": "mcq"}

    context = invocation_context(call(), "generate_questions")
    with llm.for_capability("generate_questions", context) as bound:
        answer = await bound.generate_json(PASSAGES, kind="questions", **hints)

    assert answer == await DeterministicLLM().generate_json(PASSAGES, kind="questions", **hints)


@pytest.mark.asyncio
async def test_the_model_gets_a_fixed_system_message_and_the_prompt_as_the_user(provider):
    provider("rec")
    recording = Recording()

    with GatewayLLM(gateway(recording)).for_capability(
        "chat", invocation_context(call(), "chat")
    ) as bound:
        await bound.generate("the prompt")

    [request] = recording.requests
    assert [(m.role, m.content) for m in request.messages] == [
        (ModelRole.SYSTEM, gateway_llm.TUTOR_SYSTEM),
        (ModelRole.USER, "the prompt"),
    ]
    assert request.policy is not None and request.policy.fallback_providers == []


# --- the request's context, from the verified call only (P3-4) --------------------------


@pytest.mark.asyncio
async def test_the_context_comes_from_the_verified_call(provider):
    provider("rec")
    recording = Recording()
    llm = GatewayLLM(gateway(recording))

    for capability in ("chat", "explain"):
        with llm.for_capability(capability, invocation_context(call(), capability)) as bound:
            await bound.generate("q")

    first, second = (r.context for r in recording.requests)
    assert (first.tenant_id, first.correlation_id, first.task_id) == (
        TENANT_A,
        "8f14e45f-ceea-4672-8a2b-0e3a3b4c5d6e",
        "chat",
    )
    assert second.task_id == "explain"
    # Each request is a new execution: its budget scope and its attempts' parent.
    assert first.execution_id != second.execution_id
    assert first.execution_id.startswith("exec_")


@pytest.mark.parametrize("capability", ["chat", "explain"])
def test_there_is_no_model_call_without_the_requests_context(capability):
    llm = GatewayLLM(gateway(Recording()))

    with pytest.raises(ValueError, match="needs its request's context"):
        with llm.for_capability(capability, None):
            pass


@pytest.mark.asyncio
async def test_an_unbound_gateway_llm_refuses_to_call():
    with pytest.raises(RuntimeError, match="for_capability"):
        await GatewayLLM(gateway(Recording())).generate("q")


def test_an_unknown_capability_has_no_budget_and_is_refused():
    with pytest.raises(ValueError, match="unknown capability"):
        with GatewayLLM(gateway(Recording())).for_capability(
            "summarise", invocation_context(call(), "chat")
        ):
            pass


# --- budgets per capability (P3-3) -------------------------------------------------------


def test_every_capability_has_every_limit_by_default():
    for capability in ("chat", "explain", "generate_questions"):
        limits = capability_limits(capability)
        assert limits.max_model_calls == 2
        assert limits.max_tokens and limits.max_tokens > 0
        assert limits.max_cost_usd and limits.max_cost_usd > 0


@pytest.mark.asyncio
async def test_a_request_gets_two_model_calls_at_most(provider):
    provider("rec")
    model = gateway(Recording())

    with GatewayLLM(model).for_capability("chat", invocation_context(call(), "chat")) as bound:
        await bound.generate("one")
        await bound.generate("two")
        with pytest.raises(BudgetExceededError):
            await bound.generate("three")

    # The scope ended with the request: nothing is left behind.
    assert model.budget_tracker.tracked_scope_count == 0


@pytest.mark.asyncio
async def test_a_retry_draws_on_the_requests_one_allowance(provider, monkeypatch):
    """Rule 1 with rule 6: the attempt and its retry reserve from the request's
    one allowance. A failed attempt releases its reservation (rule 6, and #49's
    test_a_failed_call_gives_its_reservation_back), so the retry can run; the
    answered call then holds the allowance, and the request has no more."""
    provider("rec")
    monkeypatch.setattr(gateway_llm.settings, "AI_BUDGET_CHAT_CALLS", 1)
    monkeypatch.setattr(
        "app.core.models.resilience.ModelRetryPolicy.compute_backoff", lambda attempt: 0.0
    )
    flaky = Recording(failures=1)

    with GatewayLLM(gateway(flaky)).for_capability(
        "chat", invocation_context(call(), "chat")
    ) as bound:
        assert await bound.generate("q") == "Answer [1]."  # failed, released, retried
        with pytest.raises(BudgetExceededError):
            await bound.generate("again")  # the allowance is spent

    assert len(flaky.requests) == 2  # the refused call never reached the provider


@pytest.mark.asyncio
async def test_attempts_are_bounded_by_the_retry_policy_not_only_the_budget(provider, monkeypatch):
    """A provider that always fails gets two attempts, the first and one retry
    (max_retries=1), whatever the budget would allow."""
    provider("rec")
    monkeypatch.setattr(gateway_llm.settings, "AI_BUDGET_CHAT_CALLS", 10)
    monkeypatch.setattr(
        "app.core.models.resilience.ModelRetryPolicy.compute_backoff", lambda attempt: 0.0
    )
    down = Recording(failures=100)

    with GatewayLLM(gateway(down)).for_capability(
        "chat", invocation_context(call(), "chat")
    ) as bound:
        with pytest.raises(ProviderUnavailableError):
            await bound.generate("q")

    assert len(down.requests) == 2


# --- failures ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_with_no_approved_provider_nothing_is_called(provider):
    provider("rec")
    unapproved = Recording()

    with GatewayLLM(gateway(unapproved, approved=set())).for_capability(
        "chat", invocation_context(call(), "chat")
    ) as bound:
        with pytest.raises(NoEligibleProviderError):
            await bound.generate("q")

    assert unapproved.requests == []


@pytest.mark.asyncio
async def test_an_unusable_structured_answer_is_a_structured_output_error(provider):
    provider("rec")
    context = invocation_context(call(), "generate_questions")

    with GatewayLLM(gateway(Recording(content="not JSON"))).for_capability(
        "generate_questions", context
    ) as bound:
        with pytest.raises(StructuredOutputError):
            await bound.generate_json("q", kind="questions", count=1)


@pytest.mark.asyncio
async def test_a_failure_is_logged_by_safe_fields_and_re_raised(provider, caplog, monkeypatch):
    provider("rec")
    monkeypatch.setattr(
        "app.core.models.resilience.ModelRetryPolicy.compute_backoff", lambda attempt: 0.0
    )
    context = invocation_context(call(), "explain")

    with caplog.at_level(logging.WARNING, logger="app.services.ai.gateway_llm"):
        with GatewayLLM(gateway(Recording(failures=5))).for_capability("explain", context) as bound:
            with pytest.raises(ProviderUnavailableError):
                await bound.generate("q")

    [line] = [r.getMessage() for r in caplog.records if "Model call failed" in r.getMessage()]
    assert "ProviderUnavailableError" in line and "capability=explain" in line
    assert f"execution={context.execution_id}" in line
    assert f"correlation={context.correlation_id}" in line
    assert MARKER not in line
