"""AI-10's exam question generator, through the model gateway (AI-01 decision P3-5).

Its model call is the gateway's: provider approval (failing closed when
none is approved), one deadline, retries, the cbt_exam budget and
sanitized errors. The request's context comes from the verified signed
call (invocation_context(call, "cbt_exam")), never from a request body.
RetrievalService scopes its passages to the verified tenant.

Not routed yet: CBTFusionEngine, its only caller, has no route.
"""

from app.core.models.contracts import ModelInvocationContext
from app.core.prompt_builder import PromptBuilder
from app.services.ai.gateway_llm import GatewayLLM
from app.services.rag.retrieval import RetrievalService


class CBTAIGenerator:
    def __init__(self, llm: GatewayLLM, retriever: RetrievalService):
        self.llm = llm
        self.retriever = retriever
        self.prompts = PromptBuilder()

    async def generate_exam_questions(
        self,
        course: str,
        topic: str,
        *,
        denied_assets: frozenset[str],
        invocation: ModelInvocationContext,
        difficulty: str = "medium",
        count: int = 5,
    ):
        # The capability is bound first: without the request's context
        # nothing is retrieved, and nothing is sent.
        with self.llm.for_capability("cbt_exam", invocation) as llm:
            context_docs = (await self.retriever.retrieve(topic, denied_assets)).items

            context_text = "\n".join(doc.get("text", "") for doc in context_docs)

            prompt = self.prompts.build(
                name="question_generator",
                topic=f"{course} - {topic}",
                count=count,
                difficulty=difficulty,
                context=context_text,
            )

            result = await llm.generate(prompt["system"] + "\n" + prompt["user"])

        return {
            "questions": result,
            "context_used": context_docs,
        }
