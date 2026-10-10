# AI-owned tables only (app/db/ownership.py). The backend owns users,
# tenants, questions, assessments, attempts and answers.
from app.db.models.learning_memory import LearningMemory as LearningMemory
from app.db.models.session_memory import SessionMemory as SessionMemory

__all__ = [
    "LearningMemory",
    "SessionMemory",
]
