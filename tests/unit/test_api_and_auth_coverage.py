import types

import pytest

from app.api.v1.ai import chat as chat_route
from app.api.v1.ai import explain as explain_route
from app.api.v1.ai import generate_questions as generate_questions_route
from app.api.v1.ai import retrieve as retrieve_route
from app.api.v1.ai import tree as tree_route
from app.core import metrics as core_metrics
from app.schemas import ai as ai_schemas
from app.schemas.common import APIResponse
from app.services.rag.recorder import SourceUsed
from app.services.rag.retrieval import Retrieved


class DummyOrchestrator:
    async def chat(self, user_id: str, question: str, denied_assets: frozenset[str]):
        return {
            "answer": "ok",
            "context_used": [],
            "memory_used": "",
            "sources_used": [{"asset_id": "a1", "run_number": 1, "attempt": 1}],
        }

    async def explain(self, subject: str, denied_assets: frozenset[str]):
        return {"explanation": f"explained {subject}", "sources_used": []}

    async def generate_questions(
        self, topic: str, denied_assets: frozenset[str], count: int = 5, **_options
    ):
        return {"questions": [], "discarded": count, "grounded": False, "sources_used": []}


# What the signed-route dependency hands a route (a verified user call).
USER_CALL = types.SimpleNamespace(
    request_id="4f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c",
    context=types.SimpleNamespace(user_id="u1"),
)


class DummyRetriever:
    async def retrieve(self, query: str, denied_assets: frozenset[str]):
        return Retrieved(
            items=[{"text": f"document for {query}"}], sources_used=(SourceUsed("a1", 1, 1),)
        )


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
        body=ai_schemas.ChatRequest(
            message="What is AI?",
            denied_assets=[],
            response_nonce="Zt3m0VbqO9aX1yR4kLw2pQ8sHn6uJc5d",
        ),
        call=USER_CALL,
        orchestrator=DummyOrchestrator(),
    )

    payload = ai_schemas.ChatResponse.model_validate_json(response.body)
    assert payload.data.answer == "ok"
    assert [s.asset_id for s in payload.sources_used] == ["a1"]


@pytest.mark.asyncio
async def test_retrieve_route_returns_api_response():
    response = await retrieve_route.retrieve(
        body=ai_schemas.RetrieveRequest(
            query="query", denied_assets=[], response_nonce="Zt3m0VbqO9aX1yR4kLw2pQ8sHn6uJc5d"
        ),
        call=USER_CALL,
        retriever=DummyRetriever(),
    )

    payload = ai_schemas.RetrieveResponse.model_validate_json(response.body)
    assert payload.data[0].text == "document for query"
    assert [s.asset_id for s in payload.sources_used] == ["a1"]


@pytest.mark.asyncio
async def test_explain_route_returns_api_response():
    response = await explain_route.explain(
        body=ai_schemas.ExplainRequest(
            subject="algebra", denied_assets=[], response_nonce="Zt3m0VbqO9aX1yR4kLw2pQ8sHn6uJc5d"
        ),
        call=USER_CALL,
        orchestrator=DummyOrchestrator(),
    )

    payload = ai_schemas.ExplainResponse.model_validate_json(response.body)
    assert payload.data.explanation == "explained algebra"
    assert payload.sources_used == []


@pytest.mark.asyncio
async def test_generate_questions_route_returns_api_response():
    response = await generate_questions_route.generate_questions(
        body=ai_schemas.GenerateQuestionsRequest(
            topic="geometry",
            count=2,
            denied_assets=[],
            response_nonce="Zt3m0VbqO9aX1yR4kLw2pQ8sHn6uJc5d",
        ),
        call=USER_CALL,
        orchestrator=DummyOrchestrator(),
    )

    payload = ai_schemas.GeneratedQuestionsResponse.model_validate_json(response.body)
    assert payload.data.questions == [] and payload.data.discarded == 2
    assert payload.sources_used == []


@pytest.mark.asyncio
async def test_tree_generate_route_returns_api_response():
    response = await tree_route.generate_tree(
        body=ai_schemas.TreeGenerateRequest(course_title="course title", topics=["topic1"]),
        call=USER_CALL,
    )
    assert isinstance(response, APIResponse)
    # route returns a TreeNode; ensure the first child has expected title
    assert response.data.children[0].title == "topic1"


def test_setup_metrics_is_a_no_op_without_the_package(monkeypatch):
    import builtins

    from fastapi import FastAPI

    real_import = builtins.__import__

    def without_instrumentator(name, *args, **kwargs):
        if name == "prometheus_fastapi_instrumentator":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    app = FastAPI()
    monkeypatch.setattr(builtins, "__import__", without_instrumentator)
    core_metrics.setup_metrics(app)
    assert all(getattr(route, "path", None) != "/metrics" for route in app.routes)


def test_setup_metrics_exposes_prometheus_metrics_on_a_real_app():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    core_metrics.setup_metrics(app)
    response = TestClient(app).get("/metrics")
    assert response.status_code == 200
    assert "# HELP" in response.text
