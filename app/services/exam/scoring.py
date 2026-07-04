class ScoringEngine:
    def grade(self, correct_answer: str, user_answer: str) -> bool:
        return correct_answer.strip().lower() == user_answer.strip().lower()

    def calculate_score(self, correct_count: int, total: int) -> float:
        if total == 0:
            return 0.0

        return (correct_count / total) * 100
