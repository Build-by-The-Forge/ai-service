import pytest

from app.services.generation.adaptive_engine import AdaptiveDifficultyEngine


class DummyMemoryRecord:
    def __init__(self, topic: str, mastery_score: float):
        self.topic = topic
        self.mastery_score = mastery_score


class DummyMemoryService:
    def __init__(self, memories):
        self._memories = memories

    async def get_user_memory(self, user_id: str):
        return self._memories


@pytest.mark.asyncio
async def test_adaptive_engine_returns_easy_for_low_mastery():
    memory_service = DummyMemoryService([DummyMemoryRecord(topic="algebra", mastery_score=0.2)])
    engine = AdaptiveDifficultyEngine(memory=memory_service)

    difficulty = await engine.get_difficulty("user-1", "algebra")

    assert difficulty == "easy"


@pytest.mark.asyncio
async def test_adaptive_engine_returns_medium_by_default():
    memory_service = DummyMemoryService([])
    engine = AdaptiveDifficultyEngine(memory=memory_service)

    difficulty = await engine.get_difficulty("user-1", "geometry")

    assert difficulty == "medium"
