import pytest

from app.services.generation.fusion_engine import CBTFusionEngine


class DummyAdaptiveEngine:
    async def get_difficulty(self, user_id: str, topic: str):
        return "medium"


class DummyGenerator:
    async def generate_exam_questions(
        self, course: str, topic: str, *, denied_assets: frozenset[str], difficulty: str, count: int
    ):
        return [{"course": course, "topic": topic, "difficulty": difficulty, "count": count}]


@pytest.mark.asyncio
async def test_generate_personalized_exam_returns_difficulty_and_exam():
    engine = CBTFusionEngine(ai_generator=DummyGenerator(), adaptive_engine=DummyAdaptiveEngine())

    result = await engine.generate_personalized_exam(
        "user-1", "math", "algebra", denied_assets=frozenset()
    )

    assert result["difficulty"] == "medium"
    assert result["exam"][0]["topic"] == "algebra"
