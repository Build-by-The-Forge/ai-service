from app.services.rag.retrieval import RetrievalService
from app.services.ai.llm import LLMProvider


class AIOrchestrator:

    def __init__(self, llm: LLMProvider, retriever: RetrievalService):
        self.llm = llm
        self.retriever = retriever

    async def chat(self, question: str):
        context_docs = await self.retriever.retrieve(question)

        context_text = "\n".join(doc.get("text", "") for doc in context_docs)

        prompt = f"""
You are a CBT AI tutor.

Use the context below to answer.

CONTEXT:
{context_text}

QUESTION:
{question}
"""

        answer = await self.llm.generate(prompt)

        return {
            "answer": answer,
            "context_used": context_docs,
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
