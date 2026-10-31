"""The learning-memory production gate: when it refuses, and what stays reachable."""

import logging
from pathlib import Path

import pytest
from prometheus_client import REGISTRY
from pydantic import ValidationError

from app import main
from app.core import learning_memory_gate as gate
from app.core.config import Settings, settings
from app.core.learning_memory_gate import LearningMemoryDisabled
from app.main import create_app
from app.services.ai.orchestrator import AIOrchestrator
from app.services.generation.adaptive_engine import AdaptiveDifficultyEngine
from app.services.memory.memory_service import MemoryService
from app.services.rag.evidence import Evidence, Passage

CALLBACK_URL = "https://api.build.example/content/ingestion/callbacks"
SECRETS = {
    "BACKEND_TO_AI_SECRET": "b2a-" + "x" * 40,
    "AI_TO_BACKEND_SECRET": "a2b-" + "z" * 40,
}
ALL_ENVIRONMENTS = ["development", "testing", "staging", "production"]
APP_DIR = Path(__file__).resolve().parents[2] / "app"


def make(environment: str, **values) -> Settings:
    return Settings(
        _env_file=None,
        APP_ENVIRONMENT=environment,
        BACKEND_CALLBACK_URL=CALLBACK_URL,
        **SECRETS,
        **values,
    )


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    monkeypatch.delenv("LEARNING_MEMORY_ENABLED", raising=False)


@pytest.fixture
def marker(monkeypatch, tmp_path):
    path = tmp_path / "production-image"
    path.touch()
    monkeypatch.setattr(gate, "PRODUCTION_IMAGE_MARKER", path)


@pytest.fixture
def no_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "PRODUCTION_IMAGE_MARKER", tmp_path / "absent")


@pytest.fixture
def closed(monkeypatch):
    monkeypatch.setattr(settings, "LEARNING_MEMORY_ENABLED", False)


class UntouchableSession:
    """A refused call must not reach the database at all."""

    def __getattr__(self, name):
        raise AssertionError(f"the database was used: {name}")


# --- startup ------------------------------------------------------------------------


@pytest.mark.parametrize("environment", ALL_ENVIRONMENTS)
def test_memory_is_off_by_default(environment, no_marker):
    assert make(environment).learning_memory_enabled is False


@pytest.mark.parametrize("environment", ["development", "testing"])
def test_development_and_testing_may_turn_it_on_outside_the_production_image(
    environment, no_marker
):
    assert make(environment, LEARNING_MEMORY_ENABLED=True).learning_memory_enabled is True


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_strict_environments_refuse_to_start_with_it_on(environment, no_marker):
    with pytest.raises(ValidationError, match="LEARNING_MEMORY_ENABLED is refused"):
        make(environment, LEARNING_MEMORY_ENABLED=True)


@pytest.mark.parametrize("environment", ALL_ENVIRONMENTS)
def test_the_production_image_refuses_whatever_the_environment_says(environment, marker):
    with pytest.raises(ValidationError, match="LEARNING_MEMORY_ENABLED is refused"):
        make(environment, LEARNING_MEMORY_ENABLED=True)


@pytest.mark.parametrize("environment", ALL_ENVIRONMENTS)
def test_the_production_image_starts_with_it_off(environment, marker):
    assert make(environment).learning_memory_enabled is False


@pytest.mark.parametrize("value", ["maybe", "", "2", "enabled-ish"])
def test_a_malformed_value_fails_startup(monkeypatch, value, no_marker):
    monkeypatch.setenv("LEARNING_MEMORY_ENABLED", value)
    with pytest.raises(ValidationError):
        make("development")


@pytest.mark.parametrize("environment", ALL_ENVIRONMENTS)
def test_opening_the_constant_lets_the_setting_decide_everywhere(monkeypatch, environment, marker):
    monkeypatch.setattr(gate, "GATE_OPEN", True)
    assert make(environment, LEARNING_MEMORY_ENABLED=True).learning_memory_enabled is True
    assert make(environment).learning_memory_enabled is False


def test_the_gate_ships_closed():
    assert gate.GATE_OPEN is False


def test_a_marker_that_cannot_be_examined_fails_closed(monkeypatch):
    class Unreadable:
        def exists(self):
            raise PermissionError("denied")

    monkeypatch.setattr(gate, "PRODUCTION_IMAGE_MARKER", Unreadable())
    with pytest.raises(PermissionError):
        gate.memory_allowed(True, "development")


def test_no_route_can_change_the_gate():
    paths = create_app(replay_guard=object()).openapi()["paths"]
    assert {p for p in paths if "memory" in p or "gate" in p} == {
        "/v1/ai/memory/query",
        "/v1/ai/memory/upsert",
    }


# --- MemoryService, the one enforcement point --------------------------------------


CALLS = [
    ("get_user_memory", ("u1",), "read"),
    ("query", ("u1", ["secret-topic"]), "read"),
    ("record", ("u1", [("secret-topic", True)]), "write"),
    ("update_mastery", ("u1", "secret-topic", 0.1), "write"),
]
CALL_IDS = [c[0] for c in CALLS]


@pytest.mark.parametrize("method,args,operation", CALLS, ids=CALL_IDS)
async def test_every_method_refuses_before_touching_the_database(
    closed, caplog, method, args, operation
):
    labels = {"operation": operation}
    name = "ai_service_learning_memory_refusals_total"
    before = REGISTRY.get_sample_value(name, labels) or 0

    with caplog.at_level(logging.DEBUG), pytest.raises(LearningMemoryDisabled):
        await getattr(MemoryService(UntouchableSession()), method)(*args)

    assert REGISTRY.get_sample_value(name, labels) == before + 1
    assert "secret-topic" not in caplog.text


@pytest.mark.parametrize("method,args,operation", CALLS, ids=CALL_IDS)
async def test_the_setting_alone_cannot_open_it_inside_the_production_image(
    marker, method, args, operation
):
    # The settings object says on, as if changed after startup: the rule still holds.
    with pytest.raises(LearningMemoryDisabled):
        await getattr(MemoryService(UntouchableSession()), method)(*args)


def test_nothing_else_in_the_application_queries_learning_memory():
    users = {
        path.relative_to(APP_DIR).as_posix()
        for path in APP_DIR.rglob("*.py")
        if any(
            needle in path.read_text(encoding="utf-8")
            for needle in (
                "select(LearningMemory",
                "insert(LearningMemory",
                "FROM learning_memory",
            )
        )
    }
    assert users == {"services/memory/memory_service.py"}


# --- callers ------------------------------------------------------------------------


class RecordingLLM:
    def __init__(self):
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return "Water moves across the membrane [1]."


class OnePassage:
    async def evidence(self, query):
        passage = Passage(
            number=1, text="Osmosis.", chunk_id="c1", document_id="d1", page_start=1, page_end=1
        )
        return Evidence(query=query, passages=(passage,))


async def test_chat_answers_without_memory_when_the_gate_is_closed(closed):
    llm = RecordingLLM()
    orchestrator = AIOrchestrator(
        llm=llm, retriever=OnePassage(), memory=MemoryService(UntouchableSession())
    )

    result = await orchestrator.chat("u1", "What is osmosis?")

    assert result["grounded"] is True
    assert result["memory_used"] == ""
    assert "mastery=" not in llm.prompts[0]


async def test_a_real_failure_reading_memory_still_fails_chat():
    # Only the gate's own refusal is expected; anything else stays an error.
    class Broken:
        async def get_user_memory(self, user_id):
            raise RuntimeError("database down")

    orchestrator = AIOrchestrator(llm=RecordingLLM(), retriever=OnePassage(), memory=Broken())
    with pytest.raises(RuntimeError):
        await orchestrator.chat("u1", "What is osmosis?")


async def test_adaptive_difficulty_falls_back_to_the_neutral_level(closed):
    engine = AdaptiveDifficultyEngine(memory=MemoryService(UntouchableSession()))
    assert await engine.get_difficulty("u1", "algebra") == "medium"


# --- visibility ---------------------------------------------------------------------


@pytest.mark.parametrize("requested,expected", [(False, 0.0), (True, 1.0)])
async def test_startup_reports_the_state_once(monkeypatch, requested, expected, no_marker):
    monkeypatch.setattr(settings, "LEARNING_MEMORY_ENABLED", requested)
    monkeypatch.setattr(main, "setup_logging", lambda: None)
    lines = []
    monkeypatch.setattr(main.logger, "info", lines.append)

    async with main.lifespan(main.create_app(replay_guard=object())):
        pass

    assert len([line for line in lines if line.startswith("Learning memory:")]) == 1
    assert REGISTRY.get_sample_value("ai_service_learning_memory_enabled") == expected
