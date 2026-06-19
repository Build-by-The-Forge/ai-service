from app.services.rag.retrieval import RetrievalService
from app.services.ai.llm import LLMProvider
from app.core.prompt_builder import PromptBuilder
from app.services.memory.memory_service import MemoryService


class AIOrchestrator:

    def __init__(self, llm: LLMProvider, retriever: RetrievalService, memory: MemoryService):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.prompts = PromptBuilder()

    async def chat(self, user_id: str, question: str):
        user_memory = await self.memory.get_user_memory(user_id)

        memory_text = "\n".join(
            f"{m.topic}: mastery={m.mastery_score}"
            for m in user_memory
        )

        context_docs = await self.retriever.retrieve(question)

        context_text = "\n".join(doc.get("text", "") for doc in context_docs)

        prompt = self.prompts.build(
            name="chat_tutor",
            context=context_text,
            question=question,
            memory=memory_text,
        )

        answer = await self.llm.generate(
            prompt["system"] + "\n" + prompt["user"]
        )

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

    async def generate_questions(self, topic: str):
        prompt = f"""
Generate 5 CBT questions on: {topic}

Include:
- multiple choice
- answers
- difficulty level
"""

        result = await self.llm.generate(prompt)

        return {"questions": result}
