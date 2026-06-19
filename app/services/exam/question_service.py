from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.models.question_bank import Question


class QuestionService:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_questions(self, course: str):
        result = await self.db.execute(
            select(Question).where(Question.course == course)
        )

        return result.scalars().all()
