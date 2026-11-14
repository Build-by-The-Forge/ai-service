import pytest

from app.core.models.contracts import ModelInvocationContext
from app.services.generation.fusion_engine import CBTFusionEngine


class DummyAdaptiveEngine:
    async def get_difficulty(self, user_id: str, topic: str):
        return "medium"


INVOCATION = ModelInvocationContext(
    tenant_id="tenant_a", execution_id="exec_1", task_id="cbt_exam", correlation_id="corr_1"
)


class DummyGenerator:
    async def generate_exam_questions(
        self,
        course: str,
        topic: str,
        *,
        denied_assets: frozenset[str],
        invocation: ModelInvocationContext,
        difficulty: str,
        count: int,
    ):
        return [
            {
                "course": course,
                "topic": topic,
                "difficulty": difficulty,
                "count": count,
                "invocation": invocation,
            }
        ]


@pytest.mark.asyncio
async def test_generate_personalized_exam_returns_difficulty_and_exam():
    engine = CBTFusionEngine(ai_generator=DummyGenerator(), adaptive_engine=DummyAdaptiveEngine())

    result = await engine.generate_personalized_exam(
        "user-1", "math", "algebra", denied_assets=frozenset(), invocation=INVOCATION
    )

    assert result["difficulty"] == "medium"
    assert result["exam"][0]["topic"] == "algebra"
    # The request's context reaches the generator, for its gateway call.
    assert result["exam"][0]["invocation"] is INVOCATION
