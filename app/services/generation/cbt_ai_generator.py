from app.core.prompt_builder import PromptBuilder
from app.services.ai.llm import LLMProvider
from app.services.rag.retrieval import RetrievalService


class CBTAIGenerator:
    def __init__(self, llm: LLMProvider, retriever: RetrievalService):
        self.llm = llm
        self.retriever = retriever
        self.prompts = PromptBuilder()

    async def generate_exam_questions(
        self,
        course: str,
        topic: str,
        difficulty: str = "medium",
        count: int = 5,
    ):
        context_docs = await self.retriever.retrieve(topic)

        context_text = "\n".join(doc.get("text", "") for doc in context_docs)

        prompt = self.prompts.build(
            name="question_generator",
            topic=f"{course} - {topic}",
            count=count,
            difficulty=difficulty,
            context=context_text,
        )

        result = await self.llm.generate(prompt["system"] + "\n" + prompt["user"])

        return {
            "questions": result,
            "context_used": context_docs,
        }
