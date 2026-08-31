from datetime import datetime

import pytest

from app.core import rate_limit
from app.core.prompts import SYSTEM_PROMPT as CORE_PROMPT
from app.services.ai.dependencies import get_ai_orchestrator
from app.services.ai.llm import LLMProvider
from app.services.ai.prompts import SYSTEM_PROMPT
from app.services.ai.router import ai_router
from app.services.ai.tenanted_retriever import TenantAwareRetrievalService
from app.services.generation.cbt_ai_generator import CBTAIGenerator
from app.services.generation.explain import Explainer
from app.services.generation.question_gen import QuestionGenerator
from app.services.memory.reward import LearningReward
from app.services.ocr.base import OCRProvider
from app.services.ocr.cleaner import TextCleaner
from app.services.ocr.openvino import OpenVINOService
from app.services.ocr.paddle import PaddleOCRService
from app.services.ocr.pdf_parser import PDFParser
from app.services.rag.chunker import Chunker
from app.services.rag.embedder import Embedder
from app.services.rag.vector_store import VectorStore
from app.services.tree.generator import TreeGeneratorService
from app.services.tree.learning_path import LearningPathService
from app.services.tree.traversal import TreeTraversalService
from app.utils.text import normalize_text
from app.utils.time import now_iso


class DummyLLM:
    async def generate(self, prompt: str) -> str:
        return f"generated:{prompt}"


class DummyRetriever:
    async def retrieve(self, query: str):
        return [{"text": f"doc for {query}"}]


class DummyDB:
    pass


@pytest.mark.asyncio
async def test_get_ai_orchestrator_constructs_orchestrator(monkeypatch):
    import app.services.ai.dependencies as ai_dependencies

    class DummyOpenAILLM:
        pass

    monkeypatch.setattr(ai_dependencies, "OpenAILLM", lambda: DummyOpenAILLM())

    orchestrator = await get_ai_orchestrator(db=DummyDB())

    assert orchestrator is not None
    assert hasattr(orchestrator, "chat")


def test_system_prompt_constants_are_defined():
    assert isinstance(SYSTEM_PROMPT, str)
    assert "AI assistant" in SYSTEM_PROMPT
    assert CORE_PROMPT == "You are an assistant for an AI service."


@pytest.mark.asyncio
async def test_llm_provider_subclass_implements_generate():
    class DummyProvider(LLMProvider):
        async def generate(self, prompt: str) -> str:
            return prompt.upper()

    provider = DummyProvider()
    output = await provider.generate("hello")

    assert output == "HELLO"


def test_ai_router_includes_routes():
    assert len(ai_router.routes) > 0


@pytest.mark.asyncio
async def test_tenant_aware_retriever_delegates_to_inner_retriever():
    service = TenantAwareRetrievalService(retriever=DummyRetriever())

    result = await service.retrieve("query")

    assert result == [{"text": "doc for query"}]


@pytest.mark.asyncio
async def test_cbt_ai_generator_returns_question_payload():
    generator = CBTAIGenerator(llm=DummyLLM(), retriever=DummyRetriever())

    result = await generator.generate_exam_questions(
        course="math",
        topic="algebra",
        difficulty="hard",
        count=2,
    )

    assert "questions" in result
    assert result["context_used"] == [{"text": "doc for algebra"}]


def test_explainer_returns_empty_string():
    assert Explainer().explain("anything") == ""


def test_question_generator_returns_empty_list():
    assert QuestionGenerator().generate("some text") == []


def test_learning_reward_returns_positive_and_negative_values():
    reward = LearningReward()

    assert reward.calculate(True, difficulty=2.0) == 0.2
    assert reward.calculate(False, difficulty=2.0) == -0.1


def test_rag_helpers_return_expected_defaults():
    assert Chunker().chunk("abc") == ["abc"]
    assert Embedder().embed("abc") == [0.0]

    vector_store = VectorStore()
    vector_store.add([0.0], {"name": "test"})
    assert vector_store.query([0.0]) == []


def test_tree_services_generate_and_navigate_tree():
    generator = TreeGeneratorService()
    root = generator.generate_base_tree("course", ["topic1", "topic2"])

    traversal = TreeTraversalService()
    assert traversal.find_node(root, root.children[0].id) is not None
    assert traversal.get_next_nodes(root, root.id) == root.children
    assert len(traversal.flatten_tree(root)) == 3

    learning_path = LearningPathService()
    assert learning_path.generate_path(root) == traversal.flatten_tree(root)
    assert learning_path.next_step(root, root.id) == root.children[0]


def test_text_and_time_utilities_work():
    assert normalize_text(" hello ") == "hello"

    timestamp = now_iso()
    assert timestamp.endswith("Z")
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def test_rate_limit_module_can_be_imported():
    assert hasattr(rate_limit, "limiter")


def test_text_cleaner_removes_newlines_and_unwanted_characters():
    cleaner = TextCleaner()

    assert cleaner.clean("Hello\nworld!!!") == "Hello world"


@pytest.mark.asyncio
async def test_ocr_provider_can_be_subclassed():
    class DummyOCR(OCRProvider):
        async def extract_text_from_image(self, image_path: str):
            return [{"text": "dummy"}]

    provider = DummyOCR()

    result = await provider.extract_text_from_image("any")
    assert result is not None


@pytest.mark.asyncio
async def test_image_ocr_providers_return_data():
    openvino = OpenVINOService()
    paddle = PaddleOCRService()

    openvino_data = await openvino.extract_text_from_image("image.png")
    assert openvino_data[0]["text"] == "openvino extracted text"

    paddle_data = await paddle.extract_text_from_image("image.png")
    assert paddle_data[0]["text"] == "sample extracted text"


def test_pdf_parser_raises_when_pymupdf_missing(monkeypatch):
    import app.services.ocr.pdf_parser as pdf_parser_module

    monkeypatch.setattr(pdf_parser_module, "fitz", None)
    parser = PDFParser()

    with pytest.raises(RuntimeError, match="PyMuPDF is required"):
        parser.extract_pages("missing.pdf")
