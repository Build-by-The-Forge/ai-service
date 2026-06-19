from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models.exam_session import ExamSession


class ExamSessionService:

    def __init__(self, db: AsyncSession):
        self.db = db

    async def start_session(self, exam_id: int, user_id: str):
        session = ExamSession(
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
