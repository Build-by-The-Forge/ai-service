from app.services.exam.session_service import ExamSessionService
from app.services.exam.question_service import QuestionService
from app.services.exam.scoring import ScoringEngine
from app.core.tenant import get_school_id


class ExamEngine:
    def __init__(self, db):
        self.session_service = ExamSessionService(db)
        self.question_service = QuestionService(db)
        self.scoring = ScoringEngine()

    async def start_exam(self, exam_id: int, user_id: str):
        school_id = get_school_id()

        session = await self.session_service.start_session(exam_id, user_id)
        questions = await self.question_service.get_questions("general", school_id=school_id)

        return {
            "session": session,
            "questions": questions,
        }
