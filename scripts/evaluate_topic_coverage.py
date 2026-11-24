"""Topic-coverage evaluation for AI-10's exam generator.

    python scripts/evaluate_topic_coverage.py [--out report.json] [--write-baseline]

Two questions, each scored on a labelled set (data/topic_coverage/v1):

- **Before generation (retrieval coverage).** For a requested course and
  topic, retrieval returns the tenant's top passages whether or not they
  cover the topic. Does the material cover the topic? 192 topics: covered
  ones (named as in the notes, paraphrased, or broad), hard same-subject
  near misses the notes don't teach, and out-of-corpus subjects.
- **After generation (question relevance).** A generated question that its
  cited passages support may still be about something other than the
  topic. Is it on the requested topic? 298 labelled (topic, question) pairs
  from 62 supported four-option questions.

Candidate signals are measured with no threshold first (the baseline); a
threshold is then chosen on the **dev** split only, by the lowest balanced
error (false-positive rate + false-negative rate; ties go to the stricter
threshold), and reported on the held-out **test** split with Wilson 95%
intervals. A false positive accepts an uncovered topic or an off-topic
question; a false negative rejects a covered topic or an on-topic question.

The retrieval side runs the production path, as scripts/evaluate_retrieval.py
does: the evaluation corpus and its past-question distractors
(data/retrieval_eval/v1) are chunked, embedded with the local model and
indexed in PgChunkIndex under a fixed evaluation tenant, and each topic's
passages come from evidence.gather with the generator's own query
("<course> <topic>"). The question side needs no index: each question cites
a page of the corpus.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import sys
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.generation.question_checks import content_terms  # noqa: E402
from app.services.ingestion.normalize import normalize_text  # noqa: E402

DATASET = ROOT / "data" / "topic_coverage" / "v1"
CORPUS = ROOT / "data" / "retrieval_eval" / "v1"
SPLITS = ("dev", "test")
EVAL_TENANT = str(uuid.uuid5(uuid.NAMESPACE_URL, "build:topic-coverage-eval"))
Z = 1.959964  # two-sided 95%


class DatasetError(Exception):
    """The labelled set is inconsistent."""


@dataclass(frozen=True)
class Dataset:
    version: str
    topics: dict[str, dict[str, Any]]
    questions: list[dict[str, Any]]
    pages: dict[str, list[str]]
    sha256: str


def _read(path: Path) -> bytes:
    # A checkout's line endings must not change the dataset's identity.
    return path.read_bytes().replace(b"\r\n", b"\n")


def load_dataset(directory: Path = DATASET, corpus: Path = CORPUS) -> Dataset:
    topics_bytes = _read(directory / "topics.json")
    questions_bytes = _read(directory / "questions.json")
    corpus_bytes = _read(corpus / "corpus.json")
    topics_doc, questions_doc = json.loads(topics_bytes), json.loads(questions_bytes)
    if topics_doc["version"] != questions_doc["version"]:
        raise DatasetError("topics and questions are different versions")
    pages = {d["id"]: d["pages"] for d in json.loads(corpus_bytes)["documents"]}

    topics: dict[str, dict[str, Any]] = {}
    for t in topics_doc["topics"]:
        if t["id"] in topics:
            raise DatasetError(f"duplicate topic id {t['id']}")
        if t["split"] not in SPLITS:
            raise DatasetError(f"{t['id']}: unknown split {t['split']}")
        if t["covered"] != t["category"].startswith("covered_"):
            raise DatasetError(f"{t['id']}: label and category disagree")
        if t["covered"]:
            e = t["evidence"]
            page = normalize_text(pages[e["document"]][e["page"] - 1])
            if normalize_text(e["span"]) not in page:
                raise DatasetError(f"{t['id']}: evidence not on {e['document']} page {e['page']}")
        elif not t.get("why"):
            raise DatasetError(f"{t['id']}: an uncovered topic says why")
        topics[t["id"]] = t

    seen: set[str] = set()
    for q in questions_doc["questions"]:
        if q["id"] in seen:
            raise DatasetError(f"duplicate question id {q['id']}")
        seen.add(q["id"])
        paired = q["on_topic"] + q["off_topic"]
        if len(set(paired)) != len(paired):
            raise DatasetError(f"{q['id']}: a topic is labelled twice")
        for tid in paired:
            if tid not in topics:
                raise DatasetError(f"{q['id']}: unknown topic {tid}")
            # Held-out means held out: a pair never mixes splits.
            if topics[tid]["split"] != q["split"]:
                raise DatasetError(f"{q['id']}: {tid} is in another split")
        for tid in q["on_topic"]:
            if not topics[tid]["covered"]:
                raise DatasetError(f"{q['id']}: on-topic for an uncovered topic {tid}")
    digest = hashlib.sha256(topics_bytes + b"\0" + questions_bytes + b"\0" + corpus_bytes)
    return Dataset(
        topics_doc["version"], topics, questions_doc["questions"], pages, digest.hexdigest()
    )


# --- signals ---------------------------------------------------------------------------


def term_coverage(terms: set[str], text: str) -> float:
    """The share of `terms` among the text's content terms; no terms, no coverage."""
    return len(terms & content_terms(text)) / len(terms) if terms else 0.0


def best_coverage(terms: set[str], texts: Iterable[str]) -> float:
    return max((term_coverage(terms, text) for text in texts), default=0.0)


def paragraphs(texts: Iterable[str]) -> list[str]:
    return [p for text in texts for p in text.split("\n\n") if p.strip()]


def question_text(item: dict[str, Any]) -> str:
    options = " ".join(str(o.get("text", "")) for o in item.get("options", []))
    return f"{item.get('prompt', '')}\n{options}\n{item.get('explanation') or ''}"


def anchor(item: dict[str, Any], cited: Sequence[str]) -> str:
    """The cited paragraph the question draws on: the one sharing most of
    the question's own content terms (first on a tie)."""
    asked = content_terms(question_text(item))
    candidates = paragraphs(cited)
    return max(candidates, key=lambda p: len(asked & content_terms(p)), default="")


def coverage_signals(topic: str, passages: Sequence[str]) -> dict[str, float]:
    terms = content_terms(topic)
    return {
        "lexical_best_passage": best_coverage(terms, passages),
        "lexical_all_passages": term_coverage(terms, "\n".join(passages)),
    }


def relevance_signals(topic: str, item: dict[str, Any], cited: Sequence[str]) -> dict[str, float]:
    terms = content_terms(topic)
    asked = question_text(item)
    return {
        "question_text": term_coverage(terms, asked),
        "cited_passage": best_coverage(terms, cited),
        "question_and_anchor": term_coverage(terms, asked + "\n" + anchor(item, cited)),
    }


# --- measurement ---------------------------------------------------------------------


def wilson(errors: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    p = errors / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def rate(errors: int, n: int) -> dict[str, Any]:
    return {
        "errors": errors,
        "n": n,
        "rate": round(errors / n, 4) if n else None,
        "ci95": wilson(errors, n),
    }


def confusion(cases: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    """cases: (positive, accepted). FP: a negative accepted; FN: a positive rejected."""
    negatives = [accepted for positive, accepted in cases if not positive]
    positives = [accepted for positive, accepted in cases if positive]
    return {
        "false_positive": rate(sum(negatives), len(negatives)),
        "false_negative": rate(sum(not a for a in positives), len(positives)),
    }


def balanced_error(table: dict[str, Any]) -> float:
    return (table["false_positive"]["rate"] or 0.0) + (table["false_negative"]["rate"] or 0.0)


@dataclass(frozen=True)
class Case:
    id: str
    split: str
    positive: bool
    group: str  # a category, for the breakdown
    signals: dict[str, float]


def thresholds(cases: Sequence[Case], signal: str) -> list[float]:
    return sorted({c.signals[signal] for c in cases if c.signals[signal] > 0})


def table(cases: Sequence[Case], accept: Callable[[Case], bool]) -> dict[str, Any]:
    return confusion([(c.positive, accept(c)) for c in cases])


def select(cases: Sequence[Case], signal: str) -> dict[str, Any]:
    """The dev split's best threshold for a signal (lowest balanced error,
    the stricter on a tie), and how it does on dev and on held-out test."""
    dev = [c for c in cases if c.split == "dev"]
    best: tuple[float, float] | None = None
    sweep = []
    for threshold in thresholds(dev, signal):
        result = table(dev, lambda c, t=threshold: c.signals[signal] >= t)
        error = balanced_error(result)
        sweep.append({"threshold": round(threshold, 4), "balanced_error": round(error, 4)})
        if best is None or error < best[0] or (error == best[0] and threshold > best[1]):
            best = (error, threshold)
    assert best is not None
    chosen = best[1]

    def accept(c: Case) -> bool:
        return c.signals[signal] >= chosen

    return {
        "threshold": round(chosen, 4),
        "dev_sweep": sweep,
        "dev": table(dev, accept),
        "test": report_split([c for c in cases if c.split == "test"], accept),
    }


def report_split(cases: Sequence[Case], accept: Callable[[Case], bool]) -> dict[str, Any]:
    groups = sorted({c.group for c in cases})
    return {
        **table(cases, accept),
        "by_category": {g: table([c for c in cases if c.group == g], accept) for g in groups},
    }


def evaluate_signals(cases: Sequence[Case], today: Callable[[Case], bool]) -> dict[str, Any]:
    """The behaviour today (no threshold), then each candidate signal with a
    threshold chosen on dev; the selected signal has the lowest dev error."""
    signals = sorted(cases[0].signals)
    candidates = {s: select(cases, s) for s in signals}
    selected = min(signals, key=lambda s: (balanced_error(candidates[s]["dev"]), signals.index(s)))
    return {
        "today": {
            split: report_split([c for c in cases if c.split == split], today) for split in SPLITS
        },
        "candidates": candidates,
        "selected": {"signal": selected, "threshold": candidates[selected]["threshold"]},
    }


def relevance_cases(dataset: Dataset) -> list[Case]:
    cases = []
    for q in dataset.questions:
        cited = [dataset.pages[p["document"]][p["page"] - 1] for p in q["passages"]]
        for tid, on in [(t, True) for t in q["on_topic"]] + [(t, False) for t in q["off_topic"]]:
            topic = dataset.topics[tid]
            group = "on_topic" if on else f"off_topic_{topic['category']}"
            cases.append(
                Case(
                    f"{q['id']}:{tid}",
                    q["split"],
                    on,
                    group,
                    relevance_signals(topic["topic"], q["item"], cited),
                )
            )
    return cases


def evaluate_relevance(dataset: Dataset) -> dict[str, Any]:
    # Today every question its passages support is kept, on topic or not.
    return evaluate_signals(relevance_cases(dataset), today=lambda c: True)


def coverage_cases(
    dataset: Dataset, passages: dict[str, list[str]], dense: dict[str, float] | None = None
) -> list[Case]:
    return [
        Case(
            tid,
            t["split"],
            t["covered"],
            t["category"],
            coverage_signals(t["topic"], passages[tid])
            | ({"dense_best_passage": dense[tid]} if dense is not None else {}),
        )
        for tid, t in dataset.topics.items()
    ]


def evaluate_coverage(
    dataset: Dataset, passages: dict[str, list[str]], dense: dict[str, float] | None = None
) -> dict[str, Any]:
    # Today the model is asked whenever retrieval returns anything.
    return evaluate_signals(
        coverage_cases(dataset, passages, dense), today=lambda c: bool(passages[c.id])
    )


def end_to_end(
    dataset: Dataset, covered: dict[str, bool], relevant: Callable[[str, dict], bool]
) -> dict[str, Any]:
    """Question pairs through both checks: an off-topic question is kept
    only if its topic passes the coverage check and it passes relevance."""
    result = {}
    for split in SPLITS:
        cases = []
        for q in dataset.questions:
            if q["split"] != split:
                continue
            for tid in q["on_topic"]:
                cases.append((True, covered[tid] and relevant(tid, q)))
            for tid in q["off_topic"]:
                cases.append((False, covered[tid] and relevant(tid, q)))
        result[split] = confusion(cases)
    return result


# --- the production path ---------------------------------------------------------------


def _sibling(name: str) -> Any:
    """The retrieval evaluation's indexing, loaded from its file (scripts/
    isn't a package)."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


async def topic_passages(dataset: Dataset, index: Any, embedder: Any, scopes: Any) -> dict:
    """Each topic's passages as the generator gets them: evidence.gather on
    "<course> <topic>" over the indexed evaluation corpus."""
    from app.services.rag.evidence import gather
    from app.services.rag.retrieval import HybridRetrievalEngine

    retrieval = _sibling("evaluate_retrieval")
    corpus = retrieval.load_dataset(CORPUS)
    tenant_id = EVAL_TENANT
    for doc_id in corpus.documents:
        await index.delete_document(tenant_id, doc_id)
    try:
        chunks = await retrieval.index_corpus(corpus, index, embedder, scopes, tenant_id)
        engine = HybridRetrievalEngine(embedder=embedder, vector_store=index, lexical_index=index)
        passages, dense = {}, {}
        for tid, t in dataset.topics.items():
            query = f"{t['course']} {t['topic']}"
            evidence = await gather(query, tenant_id, frozenset(), engine)
            passages[tid] = [p.text for p in evidence.passages]
            # For comparison only: the best cosine similarity between the
            # query and a passage it got (ADR 001 found scores overlap).
            vectors = await embedder.embed_batch([query, *passages[tid]])
            dense[tid] = round(max((cosine(vectors[0], v) for v in vectors[1:]), default=0.0), 4)
    finally:
        for doc_id in corpus.documents:
            await index.delete_document(tenant_id, doc_id)
    return {"chunks": chunks, "passages": passages, "dense": dense}


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


async def evaluate(dataset: Dataset, index: Any, embedder: Any, scopes: Any) -> dict[str, Any]:
    retrieved = await topic_passages(dataset, index, embedder, scopes)
    passages = retrieved["passages"]
    coverage = evaluate_coverage(dataset, passages, retrieved["dense"])
    relevance = evaluate_relevance(dataset)
    cov, rel = coverage["selected"], relevance["selected"]
    signals = {c.id: c.signals for c in coverage_cases(dataset, passages, retrieved["dense"])}
    covered = {tid: signals[tid][cov["signal"]] >= cov["threshold"] for tid in dataset.topics}

    def relevant(tid: str, q: dict) -> bool:
        cited = [dataset.pages[p["document"]][p["page"] - 1] for p in q["passages"]]
        signals = relevance_signals(dataset.topics[tid]["topic"], q["item"], cited)
        return signals[rel["signal"]] >= rel["threshold"]

    return {
        "dataset": {"version": dataset.version, "sha256": dataset.sha256},
        "model": getattr(embedder, "model_id", type(embedder).__name__),
        "chunks": retrieved["chunks"],
        "coverage": coverage,
        "relevance": relevance,
        "end_to_end": end_to_end(dataset, covered, relevant),
        "verdicts": {
            "coverage_rejected": sorted(tid for tid, ok in covered.items() if not ok),
        },
        # Each topic's retrieved passages, for inspection; never in the baseline.
        "passages": passages,
    }


async def _main(args: argparse.Namespace) -> int:
    from app.db.session import get_async_session_local, initialize_database, reset_engine
    from app.services.rag.onnx_embedder import OnnxMiniLMEmbedder
    from app.services.rag.pg_index import PgChunkIndex
    from app.services.rag.scopes import PgScopeStore

    dataset = load_dataset()
    await initialize_database()
    try:
        report = await evaluate(
            dataset,
            PgChunkIndex(get_async_session_local()),
            OnnxMiniLMEmbedder.from_settings(),
            PgScopeStore(get_async_session_local()),
        )
    finally:
        await reset_engine()
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
    report.pop("passages")
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.write_baseline:
        (DATASET / "baseline.json").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--write-baseline", action="store_true")
    return asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
