"""Chat, explain and question generation through the gateway (AI-01 PR 3).

GatewayLLM keeps the LLMProvider interface the orchestrator and
StructuredQuestionGenerator already use (design decision 8), so they run
through the one model execution boundary unchanged. Everything the gateway
owns applies to them: provider approval (MODEL_PROVIDERS_APPROVED, failing
closed when empty), one deadline, retries, budgets, structured output and
sanitized errors.

It sends what the old OpenAILLM sent: a fixed system message, then the
prompt as the user message. The deterministic adapter answers the last user
message, so deterministic mode gives exactly the answers it gave before.
"""

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal, cast

from app.core.config import settings
from app.core.models.budget import BudgetLimits
from app.core.models.contracts import (
    ModelInvocationContext,
    ModelMessage,
    ModelPolicy,
    ModelRequest,
    ModelResponse,
    ModelRole,
    StructuredOutput,
)
from app.core.models.errors import InvalidStructuredOutputError, ModelGatewayError
from app.core.models.gateway import ModelGateway
from app.core.service_signature_v1 import VerifiedCall
from app.services.ai.llm import LLMProvider, StructuredOutputError

logger = logging.getLogger(__name__)

Capability = Literal["chat", "explain", "generate_questions"]

TUTOR_SYSTEM = "You are an expert AI tutor."
JSON_SYSTEM = "You answer with one JSON object only."


def invocation_context(call: VerifiedCall, capability: Capability) -> ModelInvocationContext:
    """A request's model context, from the verified signed call only (decision
    P3-4): never from the request body. Each request is a new execution: the
    budget scope, and the parent of every attempt."""
    return ModelInvocationContext(
        tenant_id=call.context.tenant_id,
        execution_id=f"exec_{uuid.uuid4().hex}",
        task_id=capability,
        correlation_id=call.request_id,
    )


def capability_limits(capability: Capability) -> BudgetLimits:
    """One request's budget for the capability (decision P3-3), from settings.
    Every limit is set: none is unlimited."""
    prefix = {"chat": "CHAT", "explain": "EXPLAIN", "generate_questions": "QUESTIONS"}[capability]
    return BudgetLimits(
        max_model_calls=getattr(settings, f"AI_BUDGET_{prefix}_CALLS"),
        max_tokens=getattr(settings, f"AI_BUDGET_{prefix}_TOKENS"),
        max_cost_usd=getattr(settings, f"AI_BUDGET_{prefix}_COST_USD"),
    )


def _policy() -> ModelPolicy:
    """The configured provider only, with no fallbacks: whether it may be
    called at all is the gateway's approval check (G-33)."""
    return ModelPolicy(preferred_provider=settings.MODEL_PROVIDER, fallback_providers=[])


class GatewayLLM(LLMProvider):
    """The LLMProvider interface over the gateway. Bound to one request's
    context and capability by for_capability; unbound, it refuses to call."""

    def __init__(
        self,
        gateway: ModelGateway,
        invocation: ModelInvocationContext | None = None,
        capability: Capability | None = None,
    ) -> None:
        self._gateway = gateway
        self._invocation = invocation
        self._capability = capability

    @contextmanager
    def for_capability(
        self, capability: str, invocation: ModelInvocationContext | None
    ) -> Iterator[LLMProvider]:
        if invocation is None:
            raise ValueError("a model call needs its request's context")
        if capability not in ("chat", "explain", "generate_questions"):
            raise ValueError("unknown capability")
        bound = cast(Capability, capability)
        # The budget is the request's: set when its model work starts and
        # dropped when it ends, so nothing accumulates across requests.
        with self._gateway.budget_tracker.scope(invocation.execution_id, capability_limits(bound)):
            yield GatewayLLM(self._gateway, invocation, bound)

    async def generate(self, prompt: str) -> str:
        response = await self._call(TUTOR_SYSTEM, prompt, structured=None)
        return response.content

    async def generate_json(self, prompt: str, *, kind: str, **hints: Any) -> dict[str, Any]:
        try:
            response = await self._call(
                JSON_SYSTEM, prompt, structured=StructuredOutput(kind=kind, hints=dict(hints))
            )
        except InvalidStructuredOutputError:
            # The gateway checked the answer and it wasn't one JSON object.
            raise StructuredOutputError("model answer is not a JSON object") from None
        if response.structured is None:
            raise StructuredOutputError("model answer is not a JSON object")
        return response.structured

    async def _call(
        self, system: str, prompt: str, *, structured: StructuredOutput | None
    ) -> ModelResponse:
        if self._invocation is None or self._capability is None:
            raise RuntimeError("GatewayLLM is called through for_capability")
        request = ModelRequest(
            messages=[
                ModelMessage(role=ModelRole.SYSTEM, content=system),
                ModelMessage(role=ModelRole.USER, content=prompt),
            ],
            context=self._invocation,
            policy=_policy(),
            structured=structured,
        )
        try:
            return await self._gateway.generate(request)
        except ModelGatewayError as exc:
            # Safe structured fields only (decision P3-1): the error is the
            # gateway's own, rebuilt with fixed text, and is re-raised for the
            # route to answer 503.
            logger.warning(
                "Model call failed: %s capability=%s execution=%s correlation=%s provider=%s status=%s",
                type(exc).__name__,
                self._capability,
                self._invocation.execution_id,
                self._invocation.correlation_id,
                exc.provider,
                exc.status_code,
            )
            raise
