import types

import pytest

from app.api.v1.ai import chat as chat_route
from app.api.v1.ai import explain as explain_route
from app.api.v1.ai import generate as generate_route
from app.api.v1.ai import generate_questions as generate_questions_route
from app.api.v1.ai import retrieve as retrieve_route
from app.api.v1.ai import tree as tree_route
from app.core import metrics as core_metrics
from app.schemas import ai as ai_schemas
from app.schemas.common import APIResponse


class DummyOrchestrator:
    async def chat(self, user_id: str, question: str):
        return {"answer": "ok", "context_used": [], "memory_used": ""}

    async def explain(self, subject: str):
        return {"explanation": f"explained {subject}"}

    async def generate_questions(self, topic: str, count: int = 5, **_options):
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
