from openai import AsyncOpenAI
from app.core.config import settings
from app.services.ai.llm import LLMProvider


class OpenAILLM(LLMProvider):

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate(self, prompt: str) -> str:
        res = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert AI tutor."},
                {"role": "user", "content": prompt},
            ],
        )

        return res.choices[0].message.content
