"""Shared fixtures for AI-10's exam generator tests: a recording provider
behind a real gateway, a verified call, course evidence and a well-formed,
grounded four-option question."""

import copy
import json
import time
from typing import Any

from app.core.models.budget import BudgetTracker
from app.core.models.contracts import (
    ModelDecision,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelUsage,
)
from app.core.models.errors import ProviderUnavailableError
from app.core.models.gateway import ModelGateway
from app.core.models.provider import ModelProviderAdapter
from app.core.service_signature_v1 import VerifiedCall
from app.services.rag.evidence import Evidence, Passage
from tests.signing import TENANT_A, user_context

MARKER = "provider-said-5e2a"
PASSAGE = (
    "In cells, the mitochondrion is the organelle that makes ATP; ribosomes "
    "make proteins, and the nucleus holds the DNA."
)
PASSAGE_ONE = Passage(1, PASSAGE, "chunk-1", "asset-bio", 3, 4, run_number=2, attempt=1)

GOOD: dict[str, Any] = {
    "question_type": "mcq",
    "prompt": "Which organelle makes ATP in cells?",
    "options": [
        {"text": "Nucleus", "is_correct": False},
        {"text": "Mitochondrion", "is_correct": True},
        {"text": "Ribosome", "is_correct": False},
        {"text": "Cell wall", "is_correct": False},
    ],
    "explanation": "The mitochondrion is the organelle that makes ATP in cells.",
    "difficulty": "medium",
    "sources": [1],
}


def question(**changes: Any) -> dict[str, Any]:
    item = copy.deepcopy(GOOD)
    item.update(changes)
    return item


def call(tenant_id: str = TENANT_A) -> VerifiedCall:
    return VerifiedCall(
        sender="build-backend",
        timestamp=int(time.time()),
        request_id="5b1c9a2e-7d4f-4e8a-9c3b-1a2b3c4d5e6f",
        context=user_context(tenant_id=tenant_id, permissions=("ai.use",)),
        raw_context=b"",
    )


class Recording(ModelProviderAdapter):
    """Answers with `answer` (a dict is sent as JSON), recording every
    request; can fail first."""

    def __init__(self, answer: Any = None, failures: int = 0):
        self.answer = {"questions": [GOOD]} if answer is None else answer
        self.failures = failures
        self.requests: list[ModelRequest] = []

    @property
    def name(self) -> str:
        return "rec"

    def supports_model(self, model: str) -> bool:
        return True

    async def generate(self, request: ModelRequest, decision: ModelDecision) -> ModelResponse:
        self.requests.append(request)
        if self.failures:
            self.failures -= 1
            raise ProviderUnavailableError(f"down: {MARKER}", provider="rec")
        content = self.answer if isinstance(self.answer, str) else json.dumps(self.answer)
        return ModelResponse(
            content=content,
            model=decision.model,
            provider="rec",
            role=ModelRole.ASSISTANT,
            usage=ModelUsage(total_tokens=10),
            context=request.context,
        )

    async def generate_stream(self, request, decision):  # pragma: no cover - not used
        raise NotImplementedError
        yield


class Retriever:
    """Course evidence for any topic (or none), recording each request."""

    def __init__(self, *passages: Passage):
        self.passages = passages if passages else (PASSAGE_ONE,)
        self.calls: list[tuple[str, frozenset[str]]] = []

    async def evidence(self, query: str, denied_assets: frozenset[str]) -> Evidence:
        self.calls.append((query, denied_assets))
        # Each passage's own version, as the recorder reports it.
        sources = tuple(sorted({passage.source for passage in self.passages}))
        return Evidence(query=query, passages=self.passages, sources_used=sources)


class NoMaterial(Retriever):
    async def evidence(self, query: str, denied_assets: frozenset[str]) -> Evidence:
        self.calls.append((query, denied_assets))
        return Evidence(query=query, passages=(), sources_used=())


class Adaptive:
    async def get_difficulty(self, user_id: str, topic: str) -> str:
        return "medium"


def gateway(adapter: Recording, approved: set[str] | None = None) -> ModelGateway:
    return ModelGateway(
        providers=[adapter],
        approved_providers={"rec"} if approved is None else approved,
        budget_tracker=BudgetTracker(),
    )
