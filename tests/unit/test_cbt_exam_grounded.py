"""AI-10's question correctness and grounding boundary.

The exam generator keeps only structurally valid, four-option, grounded
questions with supported explanations, each naming the document versions it
cites. Malformed or unsupported output is discarded and counted, a topic the
material doesn't cover isn't sent to the model, and one tenant's material
never reaches another's exam. No route reaches the generator yet.
"""

import ast
from pathlib import Path

import pytest

import app.services.ai.gateway_llm as gateway_llm
from app.core import tenant
from app.core.tenant import RequestScopeMissingError
from app.services.ai.gateway_llm import GatewayLLM, invocation_context
from app.services.generation.cbt_ai_generator import CBTAIGenerator, CrossTenantInvocationError
from app.services.rag.embedder import MockEmbeddingAdapter
from app.services.rag.lexical import BM25Index
from app.services.rag.retrieval import HybridRetrievalEngine, RetrievalService
from app.services.rag.vector_store import InMemoryVectorStore
from tests.cbt_support import (
    GOOD,
    NoMaterial,
    Recording,
    Retriever,
    call,
    gateway,
    question,
)
from tests.rag_support import in_memory_engine, institutional
from tests.signing import TENANT_A, TENANT_B

APP = Path(__file__).resolve().parents[2] / "app"


@pytest.fixture(autouse=True)
def configured_provider_and_scope(monkeypatch):
    monkeypatch.setattr(gateway_llm.settings, "MODEL_PROVIDER", "rec")
    tenant.set_request_scope(TENANT_A, "user-1")
    yield
    tenant.clear_request_scope()


async def exam(answer=None, retriever=None, topic="cells", invocation_tenant=TENANT_A):
    adapter = Recording(answer)
    result = await CBTAIGenerator(
        llm=GatewayLLM(gateway(adapter)), retriever=retriever or Retriever()
    ).generate_exam_questions(
        "biology",
        topic,
        denied_assets=frozenset(),
        invocation=invocation_context(call(invocation_tenant), "cbt_exam"),
    )
    return result, adapter


@pytest.mark.asyncio
async def test_a_grounded_question_comes_back_with_the_versions_it_cites():
    result, _ = await exam()

    [kept] = result["questions"]
    assert {k: v for k, v in kept.items() if k != "citations"} == GOOD | {"sources": [1]}
    assert kept["citations"] == [
        {
            "number": 1,
            "document_id": "asset-bio",
            "page_start": 3,
            "page_end": 4,
            "run_number": 2,
            "attempt": 1,
        }
    ]
    assert result["discarded"] == 0 and result["rejected"] == {} and result["grounded"] is True
    # Every version retrieval returned, from the recorder, never the model.
    assert result["sources_used"] == [{"asset_id": "asset-bio", "run_number": 2, "attempt": 1}]
    assert (result["course"], result["topic"], result["difficulty"]) == (
        "biology",
        "cells",
        "medium",
    )


# --- malformed output --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_answer_that_is_not_json_gives_no_questions():
    result, _ = await exam("1. Which organelle makes ATP? A) Nucleus B) Mitochondrion")

    assert result["questions"] == [] and result["discarded"] == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "item",
    [
        {k: v for k, v in GOOD.items() if k != "options"},
        question(
            options=[
                {"text": "Nucleus", "is_correct": True},
                {"text": "Mitochondrion", "is_correct": True},
                {"text": "Ribosome", "is_correct": False},
                {"text": "Cell wall", "is_correct": False},
            ]
        ),
        question(extra_field="not in the shape"),
        question(difficulty="hard"),  # not what was asked for
    ],
    ids=["no options", "two correct", "unknown field", "other difficulty"],
)
async def test_a_malformed_question_is_discarded_never_repaired(item):
    result, _ = await exam({"questions": [item]})

    assert result["questions"] == [] and result["discarded"] == 1


# --- the exam rules and the support checks -----------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "item,reason",
    [
        (question(sources=[]), "untraceable"),
        (question(sources=[9]), "untraceable"),
        (
            question(prompt="Which Nobel laureate first described mitochondria in 1890?"),
            "unsupported_premise",
        ),
        (
            question(
                options=[
                    {"text": "Nucleus", "is_correct": False},
                    {"text": "Chloroplast thylakoid membrane", "is_correct": True},
                    {"text": "Ribosome", "is_correct": False},
                    {"text": "Cell wall", "is_correct": False},
                ]
            ),
            "not_answerable_from_sources",
        ),
        (
            question(
                options=[
                    {"text": "Nucleus", "is_correct": False},
                    {"text": "Mitochondrion", "is_correct": True},
                    {"text": "Ribosome", "is_correct": False},
                ]
            ),
            "not_a_four_option_mcq",
        ),
        (question(explanation=None), "missing_explanation"),
        (
            question(explanation="Mitochondria were first described in 1857."),
            "unsupported_explanation",
        ),
    ],
    ids=[
        "no sources",
        "a passage it wasn't given",
        "unsupported premise",
        "unsupported answer",
        "three options",
        "no explanation",
        "unsupported explanation",
    ],
)
async def test_an_unsupported_or_ungrounded_question_is_rejected_with_its_reason(item, reason):
    result, _ = await exam({"questions": [item, GOOD]})

    assert [q["prompt"] for q in result["questions"]] == [GOOD["prompt"]]
    assert result["discarded"] == 1 and result["rejected"] == {reason: 1}


@pytest.mark.asyncio
async def test_a_topic_the_material_doesnt_cover_is_never_sent_to_the_model():
    result, adapter = await exam(retriever=NoMaterial())

    assert (
        result["grounded"] is False and result["questions"] == [] and result["sources_used"] == []
    )
    assert adapter.requests == []


# --- tenants -----------------------------------------------------------------------------


@pytest.fixture
async def two_tenants(in_memory_scopes):
    """One engine holding a document for each tenant, each institutional for
    its own tenant: retrieval's tenant filter is the real one."""
    engine = in_memory_engine(in_memory_scopes)
    await engine.index_document(
        tenant_id=TENANT_A,
        document_id="asset-a",
        document_title="Cells",
        text="In cells, the mitochondrion is the organelle that makes ATP.",
    )
    await engine.index_document(
        tenant_id=TENANT_B,
        document_id="asset-b",
        document_title="Plants",
        text="Photosynthesis in chloroplasts turns light into chemical energy.",
    )
    await institutional(in_memory_scopes, TENANT_A, "asset-a")
    await institutional(in_memory_scopes, TENANT_B, "asset-b")
    return RetrievalService(engine=engine)


@pytest.mark.asyncio
async def test_another_tenants_material_never_reaches_the_exam(two_tenants):
    # Tenant B's topic, asked as tenant A: nothing of B's is a source or is
    # sent, whatever of A's own the search ranks.
    asked_b, adapter = await exam(retriever=two_tenants, topic="photosynthesis chloroplasts")
    assert "asset-b" not in [s["asset_id"] for s in asked_b["sources_used"]]
    for request in adapter.requests:
        sent = "\n".join(message.content for message in request.messages)
        assert "Photosynthesis" not in sent and "chloroplasts turns light" not in sent

    # Tenant A's own topic: only A's document is a source or reaches the model.
    covered, adapter = await exam(retriever=two_tenants, topic="mitochondrion ATP")
    assert [s["asset_id"] for s in covered["sources_used"]] == ["asset-a"]
    [request] = adapter.requests
    sent = "\n".join(message.content for message in request.messages)
    assert "mitochondrion" in sent and "Photosynthesis" not in sent and "asset-b" not in sent


@pytest.mark.asyncio
async def test_a_context_for_another_tenant_is_refused_before_anything_runs():
    retriever = Retriever()
    adapter = Recording()

    with pytest.raises(CrossTenantInvocationError):
        await CBTAIGenerator(
            llm=GatewayLLM(gateway(adapter)), retriever=retriever
        ).generate_exam_questions(
            "biology",
            "cells",
            denied_assets=frozenset(),
            invocation=invocation_context(call(TENANT_B), "cbt_exam"),
        )

    assert retriever.calls == [] and adapter.requests == []


@pytest.mark.asyncio
async def test_without_a_verified_scope_nothing_runs():
    tenant.clear_request_scope()
    retriever = Retriever()

    with pytest.raises(RequestScopeMissingError):
        await exam(retriever=retriever)

    assert retriever.calls == []


# --- not routed --------------------------------------------------------------------------

CBT_MODULES = {
    "app.services.generation.cbt_ai_generator",
    "app.services.generation.cbt_exam",
    "app.services.generation.fusion_engine",
}


def test_no_route_or_service_outside_generation_reaches_the_exam_generator():
    """Routing AI-10 is its own reviewed change, with its own signed
    permission: this fails until that change updates it."""
    reaching = []
    for path in sorted(APP.rglob("*.py")):
        if path.parent.name == "generation":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            module = (
                node.module
                if isinstance(node, ast.ImportFrom)
                else (
                    next((a.name for a in node.names if a.name in CBT_MODULES), None)
                    if isinstance(node, ast.Import)
                    else None
                )
            )
            if module in CBT_MODULES:
                reaching.append(f"{path.relative_to(APP.parent)} imports {module}")

    assert reaching == []


# --- the course disambiguates the topic --------------------------------------------------


class NoDense(InMemoryVectorStore):
    """Dense search off: the test embedder's vectors are hashes with no
    meaning, so ranking is the real keyword index's alone, decided by words."""

    async def query_dense(self, vector, filter, top_k=10):
        return []


@pytest.fixture
async def two_courses(in_memory_scopes):
    engine = HybridRetrievalEngine(
        embedder=MockEmbeddingAdapter(),
        vector_store=NoDense(scopes=in_memory_scopes),
        lexical_index=BM25Index(scopes=in_memory_scopes),
    )
    for asset, text in (
        (
            "asset-maths",
            "In mathematics, functions map each input in a domain to exactly one output.",
        ),
        (
            "asset-bio",
            "In biology, functions of the liver include detoxification and producing bile.",
        ),
    ):
        await engine.index_document(
            tenant_id=TENANT_A, document_id=asset, document_title=asset, text=text
        )
    await institutional(in_memory_scopes, TENANT_A, "asset-maths", "asset-bio")
    return RetrievalService(engine=engine)


@pytest.mark.asyncio
@pytest.mark.parametrize("course,first", [("mathematics", "asset-maths"), ("biology", "asset-bio")])
async def test_the_course_decides_which_material_an_overlapping_topic_draws_on(
    two_courses, course, first
):
    adapter = Recording()

    await CBTAIGenerator(
        llm=GatewayLLM(gateway(adapter)), retriever=two_courses
    ).generate_exam_questions(
        course,
        "functions",
        denied_assets=frozenset(),
        invocation=invocation_context(call(), "cbt_exam"),
    )

    [request] = adapter.requests
    sent = "\n".join(message.content for message in request.messages)
    # The course's own material is passage [1], and the prompt names the course.
    assert f"[1] ({first}," in sent
    assert f"COURSE: {course}\nTOPIC: functions" in sent
