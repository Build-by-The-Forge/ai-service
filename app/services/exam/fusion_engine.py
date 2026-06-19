from app.services.generation.cbt_ai_generator import CBTAIGenerator
from app.services.exam.adaptive_engine import AdaptiveDifficultyEngine


class CBTFusionEngine:

    def __init__(self, ai_generator: CBTAIGenerator, adaptive_engine: AdaptiveDifficultyEngine):
        self.ai_generator = ai_generator
        self.adaptive_engine = adaptive_engine

    async def generate_personalized_exam(
        self,
        user_id: str,
        course: str,
        topic: str,
    ):
        difficulty = await self.adaptive_engine.get_difficulty(user_id, topic)

        exam = await self.ai_generator.generate_exam_questions(
            course=course,
            topic=topic,
            difficulty=difficulty,
            count=10,
        )

        return {
            "difficulty": difficulty,
            "exam": exam,
        }
