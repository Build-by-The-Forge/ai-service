from app.core.learning_memory_gate import LearningMemoryDisabled
from app.core.prompt_builder import PromptBuilder
from app.services.ai.llm import LLMProvider
from app.services.generation.structured_questions import StructuredQuestionGenerator
from app.services.memory.memory_service import MemoryService
from app.services.rag.evidence import Evidence, GroundedAnswer, check_grounded
from app.services.rag.retrieval import RetrievalService


class AIOrchestrator:
    def __init__(self, llm: LLMProvider, retriever: RetrievalService, memory: MemoryService):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.prompts = PromptBuilder()

    async def chat(self, user_id: str, question: str):
        try:
            user_memory = await self.memory.get_user_memory(user_id)
        except LearningMemoryDisabled:
            # Expected while the gate is closed: chat continues without memory.
            user_memory = []

        memory_items = []
        for memory in user_memory:
            if hasattr(memory, "topic") and hasattr(memory, "mastery_score"):
                memory_items.append(f"{memory.topic}: mastery={memory.mastery_score}")
            elif isinstance(memory, dict):
                topic = memory.get("topic", "unknown")
                mastery_score = memory.get("mastery_score", 0.0)
                memory_items.append(f"{topic}: mastery={mastery_score}")

        memory_text = "\n".join(memory_items)

        evidence = await self.retriever.evidence(question)
        answer = await self._grounded(
            evidence,
            "chat_tutor",
            context=evidence.prompt_block(),
            question=question,
            memory=memory_text,
        )

        return {
            "answer": answer.text,
            "grounded": answer.grounded,
            "citations": answer.citations,
            "context_used": evidence.context(),
            "memory_used": memory_text,
        }

    async def explain(self, subject: str):
        evidence = await self.retriever.evidence(subject)
        answer = await self._grounded(
            evidence, "explain", context=evidence.prompt_block(), subject=subject
        )
        return {
            "explanation": answer.text,
            "grounded": answer.grounded,
            "citations": answer.citations,
        }

    async def _grounded(self, evidence: Evidence, template: str, **fields) -> GroundedAnswer:
        """The model answers from the evidence, or not at all: with no
        passages it is not asked, and an answer that does not cite the
        passages it was given is replaced (check_grounded)."""
        if not evidence.passages:
            return check_grounded("", evidence)
        prompt = self.prompts.build(name=template, **fields)
        return check_grounded(
            await self.llm.generate(prompt["system"] + "\n" + prompt["user"]), evidence
        )

    async def generate_questions(
        self,
        topic: str,
        count: int = 5,
        difficulty: str = "medium",
        question_type: str = "mcq",
    ):
        result = await StructuredQuestionGenerator(self.llm, self.retriever).generate(
            topic, count=count, difficulty=difficulty, question_type=question_type
        )
        return {
            "questions": [q.model_dump() for q in result.questions],
            "discarded": result.discarded,
            "grounded": result.grounded,
            "rejected": result.rejected,
        }
