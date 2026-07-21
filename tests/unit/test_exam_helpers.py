from app.services.exam.anti_cheat import AntiCheatEngine
from app.services.exam.scoring import ScoringEngine
from app.services.exam.session_state import ExamSessionState, SessionStore
from app.services.exam.timer import ExamTimer


def test_scoring_engine_grades_case_insensitively():
    engine = ScoringEngine()

    assert engine.grade("A", "a") is True
    assert engine.grade("b", "B") is True
    assert engine.grade("c", "d") is False


def test_calculate_score_handles_zero_total():
    engine = ScoringEngine()

    assert engine.calculate_score(0, 0) == 0.0
    assert engine.calculate_score(3, 5) == 60.0


def test_exam_timer_expires_after_duration(monkeypatch):
    timer = ExamTimer(duration_minutes=0)
    monkeypatch.setattr(timer, "start_time", timer.start_time - 1)

    assert timer.is_expired() is True
    assert timer.remaining_time() == 0


def test_exam_session_store_create_and_update_answer():
    store = SessionStore()
    session = store.create("s1", duration_minutes=1)

    assert store.get("s1") is session
    assert store.get_timer("s1") is not None

    store.update_answer("s1", "q1", "42")

    assert session.answers["q1"] == "42"
    assert session.answer_count == 1


def test_exam_session_state_activity_rate():
    session = ExamSessionState(session_id="s2")
    session.start_time -= 2

    session.update_answer("q1", "a")
    session.update_answer("q2", "b")

    assert session.answer_count == 2
    assert session.activity_rate() >= 0


def test_anti_cheat_detects_rapid_activity():
    engine = AntiCheatEngine()

    class DummySession:
        answers = list(range(51))
        last_activity = 1

    flags = engine.detect_abnormal_activity(DummySession())

    assert "rapid_answering" in flags
    assert "fast_switching" in flags
