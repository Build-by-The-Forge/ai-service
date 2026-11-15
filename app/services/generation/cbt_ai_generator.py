"""AI-10's exam question generator: structured, grounded and through the gateway.

Two boundaries, both required before any route may reach it:

- **Model execution (AI-01 decision P3-5).** Its one model call is the
  gateway's: provider approval (failing closed when none is approved), one
  deadline, retries, the cbt_exam budget and sanitized errors. The context
  comes from the verified signed call (invocation_context(call, "cbt_exam")),
  never from a request body, and its tenant must be the verified request's.
- **Question correctness and grounding.** Questions come from
  StructuredQuestionGenerator: the verified tenant's own passages, a JSON
  answer validated item by item, and support checks against the cited
  passages. The exam rules (cbt_exam) then require a four-option MCQ with
  a supported explanation. Anything else is discarded and counted, never
  repaired. Each kept question names the exact document versions it cites,
  and sources_used is every version retrieval returned (the retrieval
  recorder's, never the model's).

What is and isn't enforced about coverage (review of #62, issue A):

- When retrieval returns **nothing**, the model isn't asked (grounded: false).
- Retrieval returns its top passages even for a topic they don't cover,
  and retrieval scores can't tell (ADR 001), so **those passages are sent**.
  A question must then cite them and stay within them, or it is discarded:
  a question on the absent topic fails the support checks.
- **Not yet enforced:** a question that is supported by those passages but
  is about something other than the requested topic can pass. There is no
  topic-coverage check; whether to add a measured one before routing, or
  to accept this rule, is a review decision.

Not routed: CBTFusionEngine, its only caller, has no route. Routing it is
its own reviewed change, with its own signed permission.
"""

from typing import Any

from app.core import tenant
from app.core.models.contracts import ModelInvocationContext
from app.services.ai.gateway_llm import GatewayLLM
from app.services.generation.cbt_exam import check_exam_question, citations
from app.services.generation.structured_questions import StructuredQuestionGenerator
from app.services.rag.retrieval import RetrievalService


class CrossTenantInvocationError(RuntimeError):
    """The model context's tenant isn't the verified request's tenant."""


class CBTAIGenerator:
    def __init__(self, llm: GatewayLLM, retriever: RetrievalService):
        self.llm = llm
        self.retriever = retriever

    async def generate_exam_questions(
        self,
        course: str,
        topic: str,
        *,
        denied_assets: frozenset[str],
        invocation: ModelInvocationContext,
        difficulty: str = "medium",
        count: int = 5,
    ) -> dict[str, Any]:
        # The capability is bound first: without the request's context
        # nothing is retrieved, and nothing is sent.
        with self.llm.for_capability("cbt_exam", invocation) as llm:
            # Retrieval reads the verified request tenant and the model call is
            # accounted to the context's: they must be the same tenant.
            # No verified scope fails closed (RequestScopeMissingError).
            if tenant.get_tenant_id() != invocation.tenant_id:
                raise CrossTenantInvocationError("the model context's tenant isn't the request's")
            generated = await StructuredQuestionGenerator(llm, self.retriever).generate(
                topic,
                denied_assets=denied_assets,
                count=count,
                difficulty=difficulty,
                question_type="mcq",
                # The course disambiguates the topic, in retrieval and prompt.
                course=course,
            )

        questions: list[dict[str, Any]] = []
        discarded = generated.discarded
        rejected = dict(generated.rejected)
        for question in generated.questions:
            verdict = check_exam_question(question, generated.passages)
            if not verdict.ok:
                discarded += 1
                if verdict.reason:
                    rejected[verdict.reason] = rejected.get(verdict.reason, 0) + 1
                continue
            questions.append(
                {**question.model_dump(), "citations": citations(question, generated.passages)}
            )

        return {
            "course": course,
            "topic": topic,
            "difficulty": difficulty,
            "questions": questions,
            "discarded": discarded,
            "grounded": generated.grounded,
            "rejected": rejected,
            "sources_used": [source.as_dict() for source in generated.sources_used],
        }
