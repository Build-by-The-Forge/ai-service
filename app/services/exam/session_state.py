from dataclasses import dataclass, field
from typing import Dict, List
import time

from app.services.exam.timer import ExamTimer


@dataclass
class AnswerRecord:
    question_id: str
    answer: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExamSessionState:
    session_id: str
    start_time: float = field(default_factory=time.time)
    answers: Dict[str, str] = field(default_factory=dict)
    answer_log: List[AnswerRecord] = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)

    def update_answer(self, question_id: str, answer: str) -> None:
        self.answers[question_id] = answer
        self.answer_log.append(AnswerRecord(question_id=question_id, answer=answer))
        self.last_activity = time.time()

    @property
    def answer_count(self) -> int:
        return len(self.answer_log)

    def activity_rate(self) -> float:
        elapsed = time.time() - self.start_time
        return self.answer_count / max(elapsed, 1)


class SessionStore:
    def __init__(self):
        self.sessions: dict[str, ExamSessionState] = {}
        self.timers: dict[str, ExamTimer] = {}

    def get(self, session_id: str) -> ExamSessionState | None:
        return self.sessions.get(session_id)

    def create(self, session_id: str, duration_minutes: int = 30) -> ExamSessionState:
        session = ExamSessionState(session_id=session_id)
        self.sessions[session_id] = session
        self.timers[session_id] = ExamTimer(duration_minutes=duration_minutes)
        return session

    def get_timer(self, session_id: str) -> ExamTimer | None:
        return self.timers.get(session_id)

    def get_remaining_time(self, session_id: str) -> float:
        timer = self.get_timer(session_id)
        return timer.remaining_time() if timer else 0.0

    def update_answer(self, session_id: str, qid: str, answer: str) -> None:
        session = self.sessions.get(session_id)
        if session:
            session.update_answer(qid, answer)
