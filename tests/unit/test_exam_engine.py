import pytest

from app.core import tenant
from app.services.exam.exam_engine import ExamEngine


class DummySessionService:
    def __init__(self, db):
        self.db = db

    async def start_session(self, exam_id: int, user_id: str):
        return {"id": 1, "exam_id": exam_id, "user_id": user_id}


class DummyQuestionService:
    def __init__(self, db):
        self.db = db

    async def get_questions(self, category: str, school_id: int):
        return [{"question": "What is 2+2?", "school_id": school_id}]


@pytest.mark.asyncio
async def test_start_exam_uses_tenant_school_id(monkeypatch):
    monkeypatch.setattr(
        "app.services.exam.exam_engine.ExamSessionService",
        DummySessionService,
    )
    monkeypatch.setattr(
        "app.services.exam.exam_engine.QuestionService",
        DummyQuestionService,
    )

    tenant.set_school_id(999)
    engine = ExamEngine(db=None)

    result = await engine.start_exam(5, "user-abc")

    assert result["session"]["exam_id"] == 5
    assert result["session"]["user_id"] == "user-abc"
    assert result["questions"][0]["school_id"] == 999
