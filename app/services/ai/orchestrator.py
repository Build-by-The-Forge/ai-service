from app.core.prompt_builder import PromptBuilder
from app.services.ai.llm import LLMProvider
from app.services.generation.structured_questions import StructuredQuestionGenerator
from app.services.memory.memory_service import MemoryService
from app.services.rag.retrieval import RetrievalService


class AIOrchestrator:
    def __init__(self, llm: LLMProvider, retriever: RetrievalService, memory: MemoryService):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.prompts = PromptBuilder()

    async def chat(self, user_id: str, question: str):
        user_memory = await self.memory.get_user_memory(user_id)

        memory_items = []
        for memory in user_memory:
            if hasattr(memory, "topic") and hasattr(memory, "mastery_score"):
                memory_items.append(f"{memory.topic}: mastery={memory.mastery_score}")
            elif isinstance(memory, dict):
                topic = memory.get("topic", "unknown")
                mastery_score = memory.get("mastery_score", 0.0)
                memory_items.append(f"{topic}: mastery={mastery_score}")

        memory_text = "\n".join(memory_items)

        context_docs = await self.retriever.retrieve(question)

        context_text = "\n".join(doc.get("text", "") for doc in context_docs)

        prompt = self.prompts.build(
            name="chat_tutor",
            context=context_text,
            question=question,
            memory=memory_text,
        )

        answer = await self.llm.generate(prompt["system"] + "\n" + prompt["user"])

        return {
            "answer": answer,
            "context_used": context_docs,
            "memory_used": memory_text,
        }

    async def explain(self, subject: str):
        prompt = f"""
You are an expert tutoring assistant.

Explain the following subject in clear academic terms:

{subject}
"""

        answer = await self.llm.generate(prompt)

        return {"explanation": answer}

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
        }
