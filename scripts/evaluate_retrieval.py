"""Retrieval quality evaluation (P3-5): recall@k and MRR over a versioned set.

    python scripts/evaluate_retrieval.py --dataset data/retrieval_eval/v1 \
        [--out report.json] [--write-baseline]

The dataset is corpus.json (documents as lists of pages) and questions.json
(questions with expected evidence: a document, a 1-based page and a span
quoted verbatim from that page). A retrieved chunk supports a question only
if it is from the expected document, covers the expected page and contains
the evidence span, so "some chunk of the right document" is not a hit.
match=all (multi-passage questions) needs every span within the top k.
Questions of category out_of_corpus have no answer and are excluded from
recall.

The corpus goes through the production path: chunk_document, the local
embedder and PgChunkIndex (Postgres with pgvector, DATABASE_URL), under a
fresh evaluation tenant that is deleted afterwards. Three strategies are
scored separately (dense, keyword, hybrid RRF) so the hybrid is measured,
not assumed. The report records the dataset's sha256 so a result names the
exact set it came from.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.rag.contracts import RetrievalFilter, RetrievalPolicy  # noqa: E402
from app.services.ingestion.chunking import chunk_document  # noqa: E402
from app.services.ingestion.extraction import ExtractedPage  # noqa: E402
from app.services.ingestion.normalize import normalize_text  # noqa: E402

K = 5
MRR_DEPTH = 10
STRATEGIES = ("dense", "keyword", "hybrid")
NO_ANSWER = "out_of_corpus"
EVAL_TENANT = str(uuid.uuid5(uuid.NAMESPACE_URL, "build:retrieval-eval"))


class DatasetError(Exception):
    """The evaluation set is inconsistent (an evidence span is not on its page)."""


@dataclass(frozen=True)
class Dataset:
    version: str
    documents: dict[str, dict[str, Any]]
    questions: list[dict[str, Any]]
    sha256: str


def _read(path: Path) -> bytes:
    # Line endings are normalised so a checkout's conversion (CRLF on
    # Windows) does not change the dataset's identity.
    return path.read_bytes().replace(b"\r\n", b"\n")


def load_dataset(directory: Path) -> Dataset:
    corpus_bytes = _read(directory / "corpus.json")
    questions_bytes = _read(directory / "questions.json")
    distractor_bytes = _read(directory / "distractors.json")
    corpus, questions = json.loads(corpus_bytes), json.loads(questions_bytes)
    distractors = json.loads(distractor_bytes)
    if not corpus["version"] == questions["version"] == distractors["version"]:
        raise DatasetError("corpus, questions and distractors are different versions")
    documents = {d["id"]: d for d in corpus["documents"]}
    # Indexed with the corpus but never expected: competing past questions.
    for d in distractors["documents"]:
        if d["id"] in documents:
            raise DatasetError(f"distractor id {d['id']} collides with a corpus document")
        documents[d["id"]] = d
    seen: set[str] = set()
    for q in questions["questions"]:
        if q["id"] in seen:
            raise DatasetError(f"duplicate question id {q['id']}")
        seen.add(q["id"])
        if (q["category"] == NO_ANSWER) != (not q["expected"]):
            raise DatasetError(f"{q['id']}: only out_of_corpus questions have no evidence")
        for e in q["expected"]:
            pages = documents[e["document"]]["pages"]
            page = normalize_text(pages[e["page"] - 1])
            if normalize_text(e["evidence"]) not in page:
                raise DatasetError(f"{q['id']}: evidence not on {e['document']} page {e['page']}")
            if any(
                normalize_text(e["evidence"]) in normalize_text(p)
                for d in distractors["documents"]
                for p in d["pages"]
            ):
                raise DatasetError(f"{q['id']}: evidence also appears in a distractor")
    digest = hashlib.sha256(
        corpus_bytes + b"\0" + questions_bytes + b"\0" + distractor_bytes
    ).hexdigest()
    return Dataset(corpus["version"], documents, questions["questions"], digest)


def supports(chunk: Any, evidence: dict[str, Any]) -> bool:
    meta = chunk.metadata
    return (
        chunk.document_id == evidence["document"]
        and meta["page_start"] <= evidence["page"] <= meta["page_end"]
        and normalize_text(evidence["evidence"]) in normalize_text(chunk.content)
    )


def first_full_rank(question: dict[str, Any], ranked: list[Any]) -> int | None:
    """The rank (1-based) at which the question's evidence is covered."""
    needed = question["expected"]
    every = question.get("match", "any") == "all"
    found: set[int] = set()
    for rank, chunk in enumerate(ranked, start=1):
        found |= {i for i, e in enumerate(needed) if supports(chunk, e)}
        if (len(found) == len(needed)) if every else found:
            return rank
    return None


def score(questions: list[dict[str, Any]], rankings: dict[str, list[Any]]) -> dict[str, Any]:
    answerable = [q for q in questions if q["category"] != NO_ANSWER]
    ranks = {q["id"]: first_full_rank(q, rankings[q["id"]]) for q in answerable}
    hit = {qid: r is not None and r <= K for qid, r in ranks.items()}
    categories: dict[str, list[bool]] = {}
    for q in answerable:
        categories.setdefault(q["category"], []).append(hit[q["id"]])
    return {
        f"recall@{K}": round(sum(hit.values()) / len(answerable), 4),
        f"mrr@{MRR_DEPTH}": round(
            sum(1 / r for r in ranks.values() if r is not None and r <= MRR_DEPTH)
            / len(answerable),
            4,
        ),
        "by_category": {
            c: {"questions": len(v), f"recall@{K}": round(sum(v) / len(v), 4)}
            for c, v in sorted(categories.items())
        },
        "misses": sorted(qid for qid, h in hit.items() if not h),
    }


async def index_corpus(dataset: Dataset, index: Any, embedder: Any, tenant_id: str) -> int:
    total = 0
    for doc_id, doc in dataset.documents.items():
        pages = [
            ExtractedPage(number=n, kind="native", method="text_layer", text=text)
            for n, text in enumerate(doc["pages"], start=1)
        ]
        chunks = chunk_document(pages, tenant_id=tenant_id, asset_id=doc_id, run_number=1)
        vectors = await embedder.embed_batch([c.text for c in chunks])
        await index.raise_fence(tenant_id=tenant_id, asset_id=doc_id, run_number=1, attempt=1)
        total += await index.replace_asset(
            tenant_id=tenant_id,
            asset_id=doc_id,
            run_number=1,
            attempt=1,
            chunks=chunks,
            embeddings=vectors,
            embedding_model=getattr(embedder, "model_id", type(embedder).__name__),
        )
    return total


async def rank_all(dataset: Dataset, index: Any, embedder: Any, tenant_id: str) -> dict:
    from app.services.rag.retrieval import HybridRetrievalEngine

    scope = RetrievalFilter(tenant_id=tenant_id)
    engine = HybridRetrievalEngine(embedder=embedder, vector_store=index, lexical_index=index)
    policy = RetrievalPolicy(top_k=MRR_DEPTH)
    rankings: dict[str, dict[str, list[Any]]] = {s: {} for s in STRATEGIES}
    for q in dataset.questions:
        vector = await embedder.embed_text(q["question"])
        dense = await index.query_dense(vector, scope, top_k=MRR_DEPTH)
        keyword = await index.query_sparse(q["question"], scope, top_k=MRR_DEPTH)
        hybrid = await engine.retrieve(q["question"], filter=scope, policy=policy)
        rankings["dense"][q["id"]] = [c for c, _ in dense]
        rankings["keyword"][q["id"]] = [c for c, _ in keyword]
        rankings["hybrid"][q["id"]] = [rc.chunk for rc in hybrid.chunks]
    return rankings


async def evaluate(dataset: Dataset, index: Any, embedder: Any) -> dict[str, Any]:
    # A fixed tenant used only by the evaluation: chunk ids derive from the
    # tenant, and ties in ranking break on chunk id, so a fixed tenant makes
    # runs repeatable. Leftovers from an interrupted run are cleared first.
    tenant_id = EVAL_TENANT
    for doc_id in dataset.documents:
        await index.delete_document(tenant_id, doc_id)
    try:
        chunks = await index_corpus(dataset, index, embedder, tenant_id)
        rankings = await rank_all(dataset, index, embedder, tenant_id)
    finally:
        for doc_id in dataset.documents:
            await index.delete_document(tenant_id, doc_id)
    return {
        "dataset": {"version": dataset.version, "sha256": dataset.sha256},
        "model": getattr(embedder, "model_id", type(embedder).__name__),
        "chunks": chunks,
        "questions": sum(1 for q in dataset.questions if q["category"] != NO_ANSWER),
        "strategies": {s: score(dataset.questions, rankings[s]) for s in STRATEGIES},
    }


async def _main(args: argparse.Namespace) -> int:
    from app.db.session import get_async_session_local, initialize_database, reset_engine
    from app.services.rag.onnx_embedder import OnnxMiniLMEmbedder
    from app.services.rag.pg_index import PgChunkIndex

    dataset = load_dataset(args.dataset)
    await initialize_database()
    try:
        report = await evaluate(
            dataset, PgChunkIndex(get_async_session_local()), OnnxMiniLMEmbedder.from_settings()
        )
    finally:
        await reset_engine()
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    if args.write_baseline:
        (args.dataset / "baseline.json").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "retrieval_eval" / "v1")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--write-baseline", action="store_true")
    return asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
