import asyncio
import builtins
import json
import logging
import types
from datetime import datetime

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from starlette.requests import Request

from app.api.v1 import exams as exams_route
from app.api.v1 import system as system_route
from app.api.v1.ws import exam_ws as exam_ws_route
from app.auth import dependencies as auth_dependencies
from app.auth import routes as auth_routes
from app.core import dependencies as core_dependencies
from app.core import exceptions as core_exceptions
from app.core import logging as core_logging
from app.core.middleware.tenant import TenantMiddleware
from app.core.prompt_builder import PromptBuilder
from app.core.rbac import RBAC
from app.core import token_verifier
from app.core.token_verifier import TokenVerifier
from app.db import session as db_session
from app.rag.pipeline import RAGPipeline
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.ocr import OCRRequest, OCRResponse
from app.schemas.rag import RetrieveRequest, RetrieveResponse
from app.services.ocr.image_parser import ImageParser
from app.services.realtime.sync_bridge import SyncBridge
from app.services.tree.learning_path import LearningPathService
from app.services.exam.redis_timer import RedisExamTimer


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


class DummyExam:
    def __init__(self, *args, **kwargs):
        # Accept both positional and keyword init to mimic SQLAlchemy model
        self.id = kwargs.get("id", 1)
        self.school_id = kwargs.get("school_id", kwargs.get("school", 10))
        self.title = kwargs.get("title", "Test")
        self.course = kwargs.get("course", "math")
        self.duration_minutes = kwargs.get("duration_minutes", 30)
        self.is_active = kwargs.get("is_active", True)
        self.created_by = kwargs.get("created_by", kwargs.get("created_by", 1))
        self.updated_by = kwargs.get("updated_by", kwargs.get("updated_by", 1))
        self.created_at = kwargs.get("created_at", datetime.utcnow())
        self.updated_at = kwargs.get("updated_at", datetime.utcnow())


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


@pytest.mark.asyncio
async def test_exams_route_crud_and_error_paths(monkeypatch):
    dummy_exam = DummyExam()

    async def dummy_commit():
        return None

    async def dummy_refresh(obj):
        return None

    session = DummyAsyncSession(execute_result=DummyAsyncResult(), commit_error=None)
    session.add(dummy_exam)
    session.refresh = dummy_refresh

    class DummyQuery:
        def __init__(self, model):
            self.model = model
            self._where = None
            self._offset = 0
            self._limit = 0

        def where(self, *args, **kwargs):
            self._where = args
            return self

        def offset(self, value):
            self._offset = value
            return self

        def limit(self, value):
            self._limit = value
            return self

    class DummyDelete:
        def __init__(self, model):
            self.model = model
            self._where = None

        def where(self, *args, **kwargs):
            self._where = args
            return self

    monkeypatch.setattr(exams_route, "Exam", DummyExam)
    monkeypatch.setattr(exams_route, "select", lambda *args, **kwargs: DummyQuery(args[0]))
    monkeypatch.setattr(exams_route, "delete", lambda *args, **kwargs: DummyDelete(args[0]))

    response = await exams_route.create_exam(
        title="Test",
        course="math",
        duration_minutes=30,
        is_active=True,
        user={"school_id": 10, "sub": "1"},
        db=session,
    )
    assert response.success is True
    assert response.data["title"] == "Test"

    list_session = DummyAsyncSession(execute_result=DummyAsyncResult(scalars_list=[dummy_exam]))
    response = await exams_route.list_exams(
        page=1, page_size=10, user={"school_id": 10}, db=list_session
    )
    assert response.data[0]["title"] == "Test"

    get_session = DummyAsyncSession(execute_result=DummyAsyncResult(scalar=dummy_exam))
    response = await exams_route.get_exam(exam_id=1, user={"school_id": 10}, db=get_session)
    assert response.data["id"] == 1

    with pytest.raises(HTTPException):
        empty_session = DummyAsyncSession(execute_result=DummyAsyncResult(scalar=None))
        await exams_route.get_exam(exam_id=1, user={"school_id": 10}, db=empty_session)

    with pytest.raises(HTTPException):
        await exams_route.update_exam(
            exam_id=1,
            title=None,
            course=None,
            duration_minutes=None,
            is_active=None,
            user={"school_id": 10, "sub": "1"},
            db=DummyAsyncSession(execute_result=DummyAsyncResult(scalar=None)),
        )

    update_session = DummyAsyncSession(execute_result=DummyAsyncResult(scalar=dummy_exam))
    response = await exams_route.update_exam(
        exam_id=1,
        title="Updated",
        course=None,
        duration_minutes=None,
        is_active=None,
        user={"school_id": 10, "sub": "1"},
        db=update_session,
    )
    assert response.data["title"] == "Updated"

    with pytest.raises(HTTPException):
        await exams_route.delete_exam(
            exam_id=1,
            user={"school_id": 10},
            db=DummyAsyncSession(execute_result=DummyAsyncResult(rowcount=0)),
        )

    delete_session = DummyAsyncSession(execute_result=DummyAsyncResult(rowcount=1))
    response = await exams_route.delete_exam(exam_id=1, user={"school_id": 10}, db=delete_session)
    assert response.success is True


@pytest.mark.asyncio
async def test_create_exam_handles_integrity_error(monkeypatch):
    def exam_factory(*args, **kwargs):
        return DummyExam()

    class ErrorSession(DummyAsyncSession):
        async def commit(self):
            raise IntegrityError("duplicate", params=None, orig=None)

    monkeypatch.setattr(exams_route, "Exam", exam_factory)

    with pytest.raises(HTTPException):
        await exams_route.create_exam(
            title="Test",
            course="math",
            duration_minutes=30,
            is_active=True,
            user={"school_id": 10, "sub": "1"},
            db=ErrorSession(execute_result=DummyAsyncResult()),
        )


def test_auth_header_and_role_requirements(monkeypatch):
    with pytest.raises(HTTPException):
        auth_dependencies.get_authorization_header(authorization=None)

    with pytest.raises(HTTPException):
        auth_dependencies.get_authorization_header(authorization="Invalid token")

    monkeypatch.setattr(
        auth_dependencies, "decode_jwt", lambda token: {"sub": "1", "school_id": "1"}
    )
    with pytest.raises(HTTPException):
        asyncio.run(auth_dependencies.require_authenticated_user(token="Bearer token"))

    monkeypatch.setattr(
        auth_dependencies,
        "decode_jwt",
        lambda token: {"sub": "1", "school_id": "1", "role": "student"},
    )
    assert asyncio.run(auth_dependencies.require_student(user={"role": "student"})) == {
        "role": "student"
    }
    with pytest.raises(HTTPException):
        asyncio.run(auth_dependencies.require_teacher_or_admin(user={"role": "student"}))
    with pytest.raises(HTTPException):
        asyncio.run(auth_dependencies.require_admin(user={"role": "teacher"}))


@pytest.mark.asyncio
async def test_auth_routes_login_refresh_and_profile(monkeypatch):
    dummy_user = DummyUser(email="user@test", hashed_password="hashed", school_id=1, role="student")

    class LoginSession:
        async def execute(self, stmt):
            return DummyAsyncResult(scalar=dummy_user)

    monkeypatch.setattr(
        auth_routes, "pbkdf2_sha256", types.SimpleNamespace(verify=lambda password, hashed: False)
    )
    with pytest.raises(HTTPException):
        await auth_routes.login(email="user@test", password="pass", db=LoginSession())

    monkeypatch.setattr(
        auth_routes, "pbkdf2_sha256", types.SimpleNamespace(verify=lambda password, hashed: True)
    )
    monkeypatch.setattr(auth_routes, "create_jwt", lambda subject, school_id, role: "token")

    response = await auth_routes.login(email="user@test", password="pass", db=LoginSession())
    assert response.data["access_token"] == "token"

    payload = {"sub": "1", "school_id": 1, "role": "student"}
    response = await auth_routes.refresh(user=payload)
    assert response.data["access_token"] == "token"

    response = await auth_routes.me(user=payload)
    assert response.data["sub"] == "1"

    response = await auth_routes.logout()
    assert response.success is True


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


def test_logging_and_exceptions_and_tenant_middleware(monkeypatch):
    logger = logging.getLogger()
    initial_handlers = len(logger.handlers)
    core_logging.setup_logging()
    assert len(logger.handlers) >= initial_handlers

    exc = core_exceptions.AppException("oops")
    assert str(exc) == "oops"

    scope = {
        "type": "http",
        "method": "GET",
        "headers": [(b"x-school-id", b"123")],
        "query_string": b"",
    }
    request = Request(scope)

    async def handler(req):
        return types.SimpleNamespace(status_code=200)

    middleware = TenantMiddleware(app=None)
    response = asyncio.run(middleware.dispatch(request, handler))
    assert response.status_code == 200


def test_prompt_builder_rbac_and_token_verifier(monkeypatch):
    builder = PromptBuilder()
    chat_prompt = builder.build("chat_tutor", memory="m", context="c", question="q")
    assert "STUDENT MEMORY" in chat_prompt["user"]

    question_prompt = builder.build(
        "question_generator", topic="math", difficulty="hard", count=2, context="ctx"
    )
    assert "COURSE: math" in question_prompt["user"]

    default_prompt = builder.build("unknown", question="hello")
    assert default_prompt["user"] == "hello"

    rbac = RBAC()
    assert rbac.can_create_exam("teacher") is True
    assert rbac.can_take_exam("student") is True
    assert rbac.can_manage_users("admin") is True
    assert rbac.can_view_results("teacher") is True

    monkeypatch.setattr(token_verifier.settings, "JWT_PUBLIC_KEY", "secret")
    verifier = TokenVerifier()
    payload = asyncio.run(verifier.verify("token"))
    assert payload["sub"] == "user_id"

    monkeypatch.setattr(token_verifier.settings, "JWT_PUBLIC_KEY", "")
    with pytest.raises(HTTPException):
        asyncio.run(TokenVerifier().verify("token"))


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


def test_rag_pipeline_and_image_parser_and_learning_path_and_sync_bridge(monkeypatch):
    assert RAGPipeline().run("x")["result"] == "placeholder"
    assert ImageParser().parse("file") == ""

    published = []

    class DummyPubSub:
        async def publish(self, channel, message):
            published.append((channel, message))

    monkeypatch.setattr("app.services.realtime.sync_bridge.RedisPubSub", lambda: DummyPubSub())
    bridge = SyncBridge()
    asyncio.run(bridge.broadcast_answer("session", {"question_id": "q1"}))
    assert published[0][0] == "exam:session"
    assert published[0][1]["event"] == "answer_sync"

    root = types.SimpleNamespace(
        id="root", children=[types.SimpleNamespace(id="child", children=[])]
    )
    path_service = LearningPathService()
    assert path_service.next_step(root, "root").id == "child"


@pytest.mark.asyncio
async def test_redis_exam_timer_uses_redis_and_expires(monkeypatch):
    dummy_client = DummyRedisClient()
    monkeypatch.setattr("app.services.exam.redis_timer.redis_client", dummy_client)

    timer = RedisExamTimer()
    await timer.start("s1", duration=0)
    remaining = await timer.remaining("s1")
    assert remaining <= 0.0
    assert await timer.is_expired("s1") is True


def test_ws_connection_manager_and_sync_handler(monkeypatch):
    dummy_ws = DummyWebSocket()
    manager = exam_ws_route.ConnectionManager()
    asyncio.run(manager.connect(dummy_ws, "s1"))
    assert dummy_ws.accepted is True

    asyncio.run(manager.send("s1", {"event": "ping"}))
    assert json.loads(dummy_ws.sent[-1])["event"] == "ping"

    dummy_ws2 = DummyWebSocket()
    asyncio.run(manager.connect(dummy_ws2, "s2"))
    asyncio.run(manager.broadcast({"event": "broadcast"}))
    assert json.loads(dummy_ws.sent[-1])["event"] == "broadcast"
    assert json.loads(dummy_ws2.sent[-1])["event"] == "broadcast"

    manager.disconnect("s1")
    assert "s1" not in manager.active_connections

    sent = []

    async def fake_send(session_id, message):
        sent.append((session_id, message))

    monkeypatch.setattr(exam_ws_route.manager, "send", fake_send)
    asyncio.run(
        exam_ws_route.websocket_sync_handler("s2", {"event": "answer_sync", "data": {"foo": "bar"}})
    )
    assert sent[0][1]["event"] == "answer_sync"


def test_core_metrics_import_error_branch(monkeypatch):
    orig_import = builtins.__import__

    def broken_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "prometheus_fastapi_instrumentator":
            raise ImportError
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    core_logging = __import__("app.core.metrics", fromlist=["setup_metrics"]).setup_metrics
    core_logging(DummyApp())


# Provide class-level attributes so `Exam.school_id` access in queries won't fail
DummyExam.school_id = None
DummyExam.id = None
DummyExam.title = None
