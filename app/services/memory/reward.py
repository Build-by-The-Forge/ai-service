class LearningReward:

    def calculate(self, correct: bool, difficulty: float = 1.0) -> float:
        if correct:
            return 0.1 * difficulty
        return -0.05 * difficulty
