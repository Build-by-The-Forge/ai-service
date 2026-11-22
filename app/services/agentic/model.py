"""The runtime's AgentModel over the gateway's one model interface.

Each step asks for one JSON object (kind ``agent_step``) through
``ModelCaller.generate_json``, so provider approval, the deadline, retries,
budgets, structured-output checking and sanitised errors all apply as they do
for chat. No provider is named here. To run a real model later, bind a
``GatewayLLM`` (``for_capability``) to the run's invocation context and pass it
in; nothing in the runtime changes.

The interface reports no usage, so tokens are estimated from the prompt and the
answer; the gateway enforces the real budget.
"""

from app.core.agentic.model import (
    Decision,
    DecisionRequest,
    MalformedDecision,
    parse_decision,
    render_prompt,
    tokens_for,
)
from app.core.models.contracts import ModelInvocationContext
from app.services.ai.gateway_llm import ModelCaller, StructuredOutputError

STEP_KIND = "agent_step"


class ModelCallerAgentModel:
    def __init__(
        self, caller: ModelCaller, capability: str, invocation: ModelInvocationContext | None
    ) -> None:
        self._caller = caller
        self._capability = capability
        self._invocation = invocation

    async def decide(self, request: DecisionRequest) -> Decision:
        prompt = render_prompt(request)
        with self._caller.for_capability(self._capability, self._invocation) as bound:
            try:
                raw = await bound.generate_json(prompt, kind=STEP_KIND)
            except StructuredOutputError:
                raise MalformedDecision() from None
        return parse_decision(raw, tokens_for(prompt, raw))
