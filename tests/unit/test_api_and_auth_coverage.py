from datetime import UTC, datetime

import types

import pytest
from fastapi import HTTPException

from app.api.v1 import exams as exams_route
from app.api.v1.ai import chat as chat_route
from app.api.v1.ai import explain as explain_route
from app.api.v1.ai import generate as generate_route
from app.api.v1.ai import generate_questions as generate_questions_route
from app.api.v1.ai import retrieve as retrieve_route
from app.api.v1.ai import tree as tree_route
from app.auth import dependencies as auth_dependencies
from app.core import metrics as core_metrics
from app.schemas import ai as ai_schemas
from app.schemas.common import APIResponse


class DummyOrchestrator:
    async def chat(self, user_id: str, question: str):
        return {"answer": "ok", "context_used": [], "memory_used": ""}

    async def explain(self, subject: str):
        return {"explanation": f"explained {subject}"}

    async def generate_questions(self, topic: str, count: int = 5):
        return {"questions": [f"question about {topic}"] * count}


# What the signed-route dependency hands a route (a verified user call).
USER_CALL = types.SimpleNamespace(context=types.SimpleNamespace(user_id="u1"))


class DummyRetriever:
    async def retrieve(self, query: str):
        return [{"text": f"document for {query}"}]


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

    async def rollback(self):
        pass


class DummyExamModel:
    def __init__(
        self, school_id, title, course, duration_minutes, is_active, created_by, updated_by
    ):
        self.id = 1
        self.school_id = school_id
        self.title = title
        self.course = course
        self.duration_minutes = duration_minutes
        self.is_active = is_active
        self.created_by = created_by
        self.updated_by = updated_by
        self.created_at = datetime(2025, 1, 1, tzinfo=UTC)
        self.updated_at = self.created_at


@pytest.mark.asyncio
async def test_chat_route_returns_api_response():
    response = await chat_route.chat(
        body=ai_schemas.ChatRequest(message="What is AI?"),
        call=USER_CALL,
        orchestrator=DummyOrchestrator(),
    )

    assert isinstance(response, APIResponse)
    assert response.data["answer"] == "ok"


@pytest.mark.asyncio
async def test_retrieve_route_returns_api_response():
    response = await retrieve_route.retrieve(
        body=ai_schemas.RetrieveRequest(query="query"),
        call=USER_CALL,
        retriever=DummyRetriever(),
    )

    assert isinstance(response, APIResponse)
    assert response.data[0]["text"] == "document for query"


@pytest.mark.asyncio
async def test_explain_route_returns_api_response():
    response = await explain_route.explain(
        body=ai_schemas.ExplainRequest(subject="algebra"),
        call=USER_CALL,
        orchestrator=DummyOrchestrator(),
    )

    assert response.data["explanation"] == "explained algebra"


@pytest.mark.asyncio
async def test_generate_questions_route_returns_api_response():
    response = await generate_questions_route.generate_questions(
        body=ai_schemas.GenerateQuestionsRequest(topic="geometry", count=2),
        call=USER_CALL,
        orchestrator=DummyOrchestrator(),
    )

    assert response.data["questions"][0].startswith("question about geometry")
    assert len(response.data["questions"]) == 2


def test_generate_route_placeholder_returns_success():
    result = generate_route.generate()
    assert result["success"] is True


@pytest.mark.asyncio
async def test_tree_generate_route_returns_api_response():
    response = await tree_route.generate_tree(
        body=ai_schemas.TreeGenerateRequest(course_title="course title", topics=["topic1"]),
        call=USER_CALL,
    )
    assert isinstance(response, APIResponse)
    # route returns a TreeNode; ensure the first child has expected title
    assert response.data.children[0].title == "topic1"


@pytest.mark.asyncio
async def test_read_start_exam_route_uses_exam_engine(monkeypatch):
    # The start_exam API lives in the ai.exam module
    from app.api.v1.ai import exam as exam_route

    class DummyEngine:
        async def start_exam(self, exam_id: int, user_id: str):
            return {"exam_id": exam_id, "user_id": user_id}

    monkeypatch.setattr(exam_route, "ExamEngine", lambda db: DummyEngine())

    response = await exam_route.start_exam(
        exam_id=1,
        user_id="user-1",
        role="student",
        db=None,
    )

    assert response.data["exam_id"] == 1


@pytest.mark.asyncio
async def test_create_exam_route_returns_api_response(monkeypatch):
    monkeypatch.setattr(exams_route, "Exam", DummyExamModel)
    dummy_db = DummyAsyncSession()

    response = await exams_route.create_exam(
        title="Test Exam",
        course="math",
        duration_minutes=30,
        is_active=True,
        user={"school_id": 5, "sub": "11"},
        db=dummy_db,
    )

    assert response.data["title"] == "Test Exam"
    assert dummy_db.committed is True


@pytest.mark.asyncio
async def test_list_exams_route_returns_data():
    dummy_exam = DummyExamModel(5, "Test", "math", 30, True, 11, 11)
    dummy_db = DummyAsyncSession(execute_result=DummyAsyncResult(scalars_list=[dummy_exam]))

    response = await exams_route.list_exams(
        page=1,
        page_size=10,
        user={"school_id": 5},
        db=dummy_db,
    )

    assert response.data[0]["title"] == "Test"


@pytest.mark.asyncio
async def test_get_exam_route_returns_api_response():
    dummy_exam = DummyExamModel(5, "Test", "math", 30, True, 11, 11)
    dummy_db = DummyAsyncSession(execute_result=DummyAsyncResult(scalar=dummy_exam))

    response = await exams_route.get_exam(
        exam_id=1,
        user={"school_id": 5},
        db=dummy_db,
    )

    assert response.data["id"] == 1


@pytest.mark.asyncio
async def test_update_exam_route_updates_exam(monkeypatch):
    dummy_exam = DummyExamModel(5, "Test", "math", 30, True, 11, 11)
    dummy_db = DummyAsyncSession(execute_result=DummyAsyncResult(scalar=dummy_exam))

    response = await exams_route.update_exam(
        exam_id=1,
        title="Updated",
        course=None,
        duration_minutes=None,
        is_active=None,
        user={"school_id": 5, "sub": "11"},
        db=dummy_db,
    )

    assert response.data["title"] == "Updated"


@pytest.mark.asyncio
async def test_delete_exam_route_commits_and_returns_success():
    dummy_db = DummyAsyncSession(execute_result=DummyAsyncResult(rowcount=1))

    response = await exams_route.delete_exam(
        exam_id=1,
        user={"school_id": 5},
        db=dummy_db,
    )

    assert response.success is True


def test_get_authorization_header_raises_without_bearer():
    with pytest.raises(HTTPException) as exc:
        auth_dependencies.get_authorization_header(authorization=None)

    assert exc.value.detail == "Authorization header missing or malformed"


def test_get_authorization_header_parses_bearer_token():
    token = auth_dependencies.get_authorization_header(authorization="Bearer abc123")
    assert token == "abc123"


@pytest.mark.asyncio
async def test_require_authenticated_user_sets_school_id(monkeypatch):
    monkeypatch.setattr(
        auth_dependencies,
        "decode_jwt",
        lambda token: {"sub": "1", "school_id": "42", "role": "student"},
    )

    payload = await auth_dependencies.require_authenticated_user(token="Bearer token")
    assert payload["school_id"] == "42"


@pytest.mark.asyncio
async def test_role_requirements_call_rbac(monkeypatch):
    monkeypatch.setattr(auth_dependencies.rbac, "can_take_exam", lambda role: True)
    monkeypatch.setattr(auth_dependencies.rbac, "can_create_exam", lambda role: True)
    monkeypatch.setattr(auth_dependencies.rbac, "can_manage_users", lambda role: True)

    user = {"role": "admin"}
    assert await auth_dependencies.require_student(user=user) == user
    assert await auth_dependencies.require_teacher_or_admin(user=user) == user
    assert await auth_dependencies.require_admin(user=user) == user


def test_setup_metrics_handles_missing_package():
    # Provide a minimal dummy app object so instrumentation can attach safely
    class DummyApp:
        def add_middleware(self, *args, **kwargs):
            return None

        def get(self, endpoint, include_in_schema=True, tags=None, **kwargs):
            # Return a decorator that accepts a function and returns it unchanged
            def decorator(func):
                return func

            return decorator

    core_metrics.setup_metrics(app=DummyApp())
