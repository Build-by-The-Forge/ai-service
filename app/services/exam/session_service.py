from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models.exam_session import ExamSession
from app.db.models.exam import Exam
from app.core.tenant import get_school_id


class ExamSessionService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def start_session(self, exam_id: int, user_id: str):
        school_id = get_school_id()

        result = await self.db.execute(
            select(Exam).where(
                Exam.id == exam_id,
                Exam.school_id == school_id,
            )
        )
        exam = result.scalar_one_or_none()

        if not exam:
            raise ValueError("Exam not found for current school")

        session = ExamSession(
            school_id=school_id,
            exam_id=exam_id,
            user_id=user_id,
            status="active",
            current_index=0,
            score=0.0,
        )

        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)

        return session
