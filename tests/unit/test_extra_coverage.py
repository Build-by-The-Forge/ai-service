import asyncio
import json
import sys
import types

import pytest

from app.api.v1.ai import ocr as ocr_route
from app.api.v1.ai import read as read_route
from app.api.v1.ai import retrieve as retrieve_route
from app.core import metrics as core_metrics
from app.schemas.ai import ReadStartRequest, RetrieveRequest
from app.schemas.tree import TreeNode
from app.services.memory.memory_service import MemoryService
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


def test_ocr_route_placeholder_returns_success():
    result = ocr_route.ocr()
    assert result["success"] is True
    assert "placeholder" in result["message"].lower()


@pytest.mark.asyncio
async def test_retrieve_route_returns_api_response_with_dummy_retriever():
    class DummyRetriever:
        async def retrieve(self, query: str):
            return [{"text": f"dummy {query}"}]

    response = await retrieve_route.retrieve(
        body=RetrieveRequest(query="test"), call=None, retriever=DummyRetriever()
    )

    assert response.success is True
    assert response.data[0]["text"] == "dummy test"


def test_start_reading_route_generates_path():
    root = TreeNode(
        id="root",
        title="Root Course",
        level=0,
        children=[TreeNode(id="topic1", title="Topic 1", level=1)],
    )
    response = asyncio.run(read_route.start_reading(body=ReadStartRequest(root=root), call=None))

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
async def test_memory_service_get_user_memory_and_update_mastery(monkeypatch):
    monkeypatch.setattr("app.services.memory.memory_service.get_tenant_id", lambda: "77")

    dummy_record = types.SimpleNamespace(
        tenant_id="77",
        user_id="u1",
        topic="math",
        mastery_score=0.5,
        attempts=1,
        last_interaction="",
    )
    dummy_db = DummyAsyncSession(execute_result=DummyResult(scalars_list=[dummy_record]))

    memory_service = MemoryService(db=dummy_db)
    records = await memory_service.get_user_memory(user_id="u1")
    assert records == [dummy_record]

    dummy_update_db = DummyAsyncSession(execute_result=DummyResult(scalar=None))
    updated = await MemoryService(db=dummy_update_db).update_mastery(
        user_id="u1", topic="math", delta=0.1
    )

    assert updated.tenant_id == "77"
    assert updated.mastery_score == 0.1
    assert updated.attempts == 1
    assert dummy_update_db.committed is True


def test_pdf_parser_raises_when_pymupdf_is_missing(monkeypatch):
    monkeypatch.setattr("app.services.ocr.pdf_parser.fitz", None)

    parser = PDFParser()
    with pytest.raises(RuntimeError, match="PyMuPDF is required"):
        parser.extract_pages("dummy.pdf")
