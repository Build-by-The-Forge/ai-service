import asyncio
import json
import types
import sys

import pytest

from app.api.v1.ai import exam_submit as exam_submit_route
from app.api.v1.ai import fusion_exam as fusion_exam_route
from app.api.v1.ai import ocr as ocr_route
from app.api.v1.ai import retrieve as retrieve_route
from app.api.v1.ai import read as read_route
from app.core import metrics as core_metrics
from app.schemas.tree import TreeNode
from app.services.exam.redis_session_store import RedisSessionStore
from app.services.exam.session_service import ExamSessionService
from app.services.memory.memory_service import MemoryService
from app.services.realtime.pubsub import RedisPubSub
from app.services.ocr.pdf_parser import PDFParser


class DummyResult:
    def __init__(self, scalar=None, scalars_list=None, rowcount=0):
        self._scalar = scalar
        self._scalars_list = scalars_list or []
        self.rowcount = rowcount

    def scalar_one_or_none(self):
        return self._scalar

    def scalars(self):
        return self

    def all(self):
        return self._scalars_list


class DummyAsyncSession:
    def __init__(self, execute_result=None):
        self.execute_result = execute_result
        self.added = []
        self.committed = False
        self.refreshed = []

    async def execute(self, stmt):
        return self.execute_result

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.committed = True

    async def refresh(self, obj):
        self.refreshed.append(obj)


class DummyRedisClient:
    def __init__(self):
        self.storage = {}
        self.published = []

    async def set(self, key, value):
        self.storage[key] = value

    async def get(self, key):
        return self.storage.get(key)

    async def publish(self, channel, message):
        self.published.append((channel, message))

    def pubsub(self, ignore_subscribe_messages=True):
        return DummyPubSubHandler()


class DummyPubSubHandler:
    def __init__(self):
        self.subscribed_channels = []

    async def subscribe(self, channel):
        self.subscribed_channels.append(channel)

    async def listen(self):
        yield {
            "type": "message",
            "data": json.dumps({"hello": "world"}).encode("utf-8"),
        }
        yield {"type": "message", "data": json.dumps({"foo": "bar"})}


class DummyExam:
    def __init__(self, school_id):
        self.id = 1
        self.school_id = school_id


@pytest.mark.asyncio
async def test_submit_answer_route_grades_correct_and_incorrect_answers():
    correct_response = await exam_submit_route.submit_answer(
        correct_answer="42",
        user_answer="42",
        role="student",
        db=None,
    )
    incorrect_response = await exam_submit_route.submit_answer(
        correct_answer="42",
        user_answer="24",
        role="student",
        db=None,
    )

    assert correct_response.success is True
    assert correct_response.data["is_correct"] is True
    assert incorrect_response.data["is_correct"] is False


@pytest.mark.asyncio
async def test_generate_ai_exam_route_uses_engine_and_returns_payload(monkeypatch):
    class DummyFusionEngine:
        def __init__(self, ai_generator, adaptive_engine):
            self.ai_generator = ai_generator
            self.adaptive_engine = adaptive_engine

        async def generate_personalized_exam(self, user_id, course, topic):
            return {"difficulty": "easy", "exam": {"course": course, "topic": topic}}

    monkeypatch.setattr(fusion_exam_route, "CBTFusionEngine", DummyFusionEngine)
    monkeypatch.setattr(fusion_exam_route, "OpenAILLM", lambda: object())
    monkeypatch.setattr(fusion_exam_route, "TenantAwareRetrievalService", lambda: object())
    monkeypatch.setattr(fusion_exam_route, "MemoryService", lambda db: object())

    response = await fusion_exam_route.generate_ai_exam(
        user_id="user-1",
        course="history",
        topic="world",
        role="admin",
        db=None,
    )

    assert response.success is True
    assert response.data["difficulty"] == "easy"
    assert response.data["exam"]["topic"] == "world"


def test_ocr_route_placeholder_returns_success():
    result = ocr_route.ocr()
    assert result["success"] is True
    assert "placeholder" in result["message"].lower()


@pytest.mark.asyncio
async def test_retrieve_route_returns_api_response_with_dummy_retriever():
    class DummyRetriever:
        async def retrieve(self, query: str):
            return [{"text": f"dummy {query}"}]

    response = await retrieve_route.retrieve(query="test", retriever=DummyRetriever())

    assert response.success is True
    assert response.data[0]["text"] == "dummy test"


def test_start_reading_route_generates_path():
    root = TreeNode(
        id="root",
        title="Root Course",
        level=0,
        children=[TreeNode(id="topic1", title="Topic 1", level=1)],
    )
    response = asyncio.run(read_route.start_reading(root=root))

    assert response.success is True
    assert len(response.data) == 2
    assert response.data[0].title == "Root Course"


def test_setup_metrics_instrumentator_is_used(monkeypatch):
    class DummyInstrumentator:
        def __init__(self):
            self.instrumented = False
            self.exposed = False

        def instrument(self, app):
            self.instrumented = True
            return self

        def expose(self, app):
            self.exposed = True
            return app

    dummy_module = types.SimpleNamespace(Instrumentator=DummyInstrumentator)
    monkeypatch.setitem(sys.modules, "prometheus_fastapi_instrumentator", dummy_module)

    class DummyApp:
        pass

    core_metrics.setup_metrics(app=DummyApp())


@pytest.mark.asyncio
async def test_redis_session_store_create_get_and_update_answer(monkeypatch):
    dummy_client = DummyRedisClient()
    monkeypatch.setattr(
        "app.services.exam.redis_session_store.redis_client",
        dummy_client,
    )

    store = RedisSessionStore()
    await store.create_session("s1", {"user_id": "u1", "answers": {"q1": "a"}})
    stored = await store.get_session("s1")
    assert stored["user_id"] == "u1"
    assert stored["answers"]["q1"] == "a"

    updated = await store.update_answer("s1", "q2", "b")
    assert updated["answers"]["q2"] == "b"
    assert updated["last_activity"] >= 0


@pytest.mark.asyncio
async def test_redis_session_store_update_answer_returns_none_when_missing(monkeypatch):
    import app.services.exam.redis_session_store as redis_session_store_module

    dummy_client = DummyRedisClient()
    monkeypatch.setattr(redis_session_store_module, "redis_client", dummy_client)

    store = RedisSessionStore()
    result = await store.update_answer("unknown", "q1", "a")

    assert result is None


@pytest.mark.asyncio
async def test_exam_session_service_start_session_with_and_without_exam(monkeypatch):
    monkeypatch.setattr("app.services.exam.session_service.get_school_id", lambda: 100)

    service = ExamSessionService(
        db=DummyAsyncSession(execute_result=DummyResult(scalar=DummyExam(school_id=100)))
    )
    session = await service.start_session(exam_id=1, user_id="user-1")

    assert session.exam_id == 1
    assert session.user_id == "user-1"
    assert session.status == "active"

    with pytest.raises(ValueError):
        service_missing = ExamSessionService(
            db=DummyAsyncSession(execute_result=DummyResult(scalar=None))
        )
        await service_missing.start_session(exam_id=1, user_id="user-1")


@pytest.mark.asyncio
async def test_memory_service_get_user_memory_and_update_mastery(monkeypatch):
    monkeypatch.setattr("app.services.memory.memory_service.get_school_id", lambda: 77)

    dummy_record = types.SimpleNamespace(
        school_id=77, user_id="u1", topic="math", mastery_score=0.5, attempts=1, last_interaction=""
    )
    dummy_db = DummyAsyncSession(execute_result=DummyResult(scalars_list=[dummy_record]))

    memory_service = MemoryService(db=dummy_db)
    records = await memory_service.get_user_memory(user_id="u1")
    assert records == [dummy_record]

    dummy_update_db = DummyAsyncSession(execute_result=DummyResult(scalar=None))
    updated = await MemoryService(db=dummy_update_db).update_mastery(
        user_id="u1", topic="math", delta=0.1
    )

    assert updated.mastery_score == 0.1
    assert updated.attempts == 1
    assert dummy_update_db.committed is True


def test_pdf_parser_raises_when_pymupdf_is_missing(monkeypatch):
    monkeypatch.setattr("app.services.ocr.pdf_parser.fitz", None)

    parser = PDFParser()
    with pytest.raises(RuntimeError, match="PyMuPDF is required"):
        parser.extract_pages("dummy.pdf")


@pytest.mark.asyncio
async def test_pubsub_publish_and_subscribe(monkeypatch):
    dummy_client = DummyRedisClient()
    monkeypatch.setattr(
        "app.services.realtime.pubsub.redis_client",
        dummy_client,
    )

    published_messages = []

    async def handler(message):
        published_messages.append(message)

    pubsub = RedisPubSub()
    await pubsub.publish("channel", {"hello": "world"})
    assert dummy_client.published == [("channel", json.dumps({"hello": "world"}))]

    await pubsub.subscribe("channel", handler)
    assert published_messages == [{"hello": "world"}, {"foo": "bar"}]
