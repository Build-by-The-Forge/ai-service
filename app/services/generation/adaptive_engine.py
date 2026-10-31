from app.core.learning_memory_gate import LearningMemoryDisabled
from app.services.memory.memory_service import MemoryService


class AdaptiveDifficultyEngine:
    def __init__(self, memory: MemoryService):
        self.memory = memory

    async def get_difficulty(self, user_id: str, topic: str) -> str:
        try:
            memories = await self.memory.get_user_memory(user_id)
        except LearningMemoryDisabled:
            # Treated as a student with no memory of the topic.
            return "medium"

        for m in memories:
            if m.topic == topic:
                score = m.mastery_score

                if score < 0.3:
                    return "easy"
                elif score < 0.7:
                    return "medium"
                else:
                    return "hard"

        return "medium"
