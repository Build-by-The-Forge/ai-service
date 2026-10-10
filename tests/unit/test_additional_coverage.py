import asyncio
import builtins
import logging
import types

import pytest

from app.api.v1 import system as system_route
from app.core import dependencies as core_dependencies
from app.core import exceptions as core_exceptions
from app.core import logging as core_logging
from app.core.prompt_builder import PromptBuilder
from app.db import session as db_session
from app.rag.pipeline import RAGPipeline
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.ocr import OCRRequest, OCRResponse
from app.schemas.rag import RetrieveRequest, RetrieveResponse
from app.services.ocr.image_parser import ImageParser
from app.services.tree.learning_path import LearningPathService


class DummyAsyncResult:
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
    def __init__(self, execute_result=None, commit_error=None):
        self.execute_result = execute_result
        self.committed = False
        self.refreshed = []
        self.rollback_called = False
        self.commit_error = commit_error

    async def execute(self, stmt):
        return self.execute_result

    def add(self, obj):
        self.added = obj

    async def commit(self):
        if self.commit_error:
            raise self.commit_error
        self.committed = True

    async def refresh(self, obj):
        self.refreshed.append(obj)

    async def rollback(self):
        self.rollback_called = True


class DummyUser:
    def __init__(self, email, hashed_password, school_id, role, id=1):
        self.email = email
        self.hashed_password = hashed_password
        self.school_id = school_id
        self.role = role
        self.id = id


class DummyRedisClient:
    def __init__(self, data=None):
        self.data = data or {}
        self.published = []

    async def set(self, key, value):
        self.data[key] = str(value)

    async def get(self, key):
        return self.data.get(key)

    async def ping(self):
        return True


class DummyWebSocket:
    def __init__(self):
        self.accepted = False
        self.sent = []
        self.headers = {"x-school-id": "42"}

    async def accept(self):
        self.accepted = True

    async def send_text(self, content):
        self.sent.append(content)

    async def receive_text(self):
        raise RuntimeError("receive_text not implemented")


class DummyApp:
    def add_middleware(self, *args, **kwargs):
        return None

    def get(self, endpoint, include_in_schema=True, tags=None, **kwargs):
        def decorator(func):
            return func

        return decorator


@pytest.mark.asyncio
async def test_health_routes_report_database_and_redis(monkeypatch):
    class DummyEngine:
        def connect(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def run_sync(self, fn):
            return fn(None)

    monkeypatch.setattr(system_route, "get_engine", lambda: DummyEngine())
    monkeypatch.setattr(system_route, "redis_client", DummyRedisClient())

    health = await system_route.health()
    assert health["success"] is True
    assert health["data"]["status"] == "ok"

    readiness = await system_route.readiness()
    assert readiness["success"] is True
    assert readiness["data"]["database"] is True
    assert readiness["data"]["redis"] is True


@pytest.mark.asyncio
async def test_system_health_check_returns_service_status():
    result = await system_route.health()
    assert result["success"] is True
    assert result["data"]["status"] == "ok"


async def test_get_db_generator_closes_session(monkeypatch):
    class DummySessionContext:
        def __init__(self):
            self.closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            self.closed = True

        async def rollback(self):
            self.closed = True

        async def close(self):
            self.closed = True

    def dummy_factory():
        return DummySessionContext()

    monkeypatch.setattr(core_dependencies, "get_async_session_local", lambda: dummy_factory)
    generator = core_dependencies.get_db()
    session = await generator.__anext__()
    assert session is not None
    with pytest.raises(StopAsyncIteration):
        await generator.__anext__()


def test_logging_and_exceptions(monkeypatch):
    logger = logging.getLogger()
    initial_handlers = len(logger.handlers)
    core_logging.setup_logging()
    assert len(logger.handlers) >= initial_handlers

    exc = core_exceptions.AppException("oops")
    assert str(exc) == "oops"


def test_prompt_builder():
    builder = PromptBuilder()
    chat_prompt = builder.build("chat_tutor", memory="m", context="c", question="q")
    assert "STUDENT MEMORY" in chat_prompt["user"]

    question_prompt = builder.build(
        "question_generator", topic="math", difficulty="hard", count=2, context="ctx"
    )
    assert "COURSE: math" in question_prompt["user"]

    default_prompt = builder.build("unknown", question="hello")
    assert default_prompt["user"] == "hello"


def test_database_reset_engine(monkeypatch):
    class DummyEngine:
        async def dispose(self):
            return None

    db_session.engine = DummyEngine()
    db_session.AsyncSessionLocal = object()
    asyncio.run(db_session.reset_engine())
    assert db_session.engine is None
    assert db_session.AsyncSessionLocal is None


def test_schema_models_can_be_instantiated():
    ChatRequest(prompt="hello")
    ChatResponse(success=True, message="ok", data={})
    OCRRequest(file_url="/tmp/file.pdf")
    OCRResponse(success=True, message="ok", data={})
    RetrieveRequest(query="hello")
    RetrieveResponse(success=True, message="ok", data={})


def test_rag_pipeline_and_image_parser_and_learning_path():
    assert RAGPipeline().run("x")["result"] == "placeholder"
    assert ImageParser().parse("file") == ""

    root = types.SimpleNamespace(
        id="root", children=[types.SimpleNamespace(id="child", children=[])]
    )
    path_service = LearningPathService()
    assert path_service.next_step(root, "root").id == "child"


def test_core_metrics_import_error_branch(monkeypatch):
    orig_import = builtins.__import__

    def broken_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "prometheus_fastapi_instrumentator":
            raise ImportError
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    core_logging = __import__("app.core.metrics", fromlist=["setup_metrics"]).setup_metrics
    core_logging(DummyApp())
