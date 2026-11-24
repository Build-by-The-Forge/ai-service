"""The topic-coverage gate (AI-10): the coverage check, run on the passages
production retrieval returns, must not regress below the committed baseline
(data/topic_coverage/v1/baseline.json), and must decide exactly as the
evaluated signal does.

The evaluation runs the production path (chunker, local embedder,
PgChunkIndex, evidence.gather with the generator's query). The tolerance is
one topic per error class and split, for floating-point differences in
ranking between machines. Changing the check or the set means re-measuring
and committing a new baseline (scripts/evaluate_topic_coverage.py
--write-baseline) with the change.

Needs PostgreSQL with pgvector (DATABASE_URL) and the verified model files
(EMBEDDING_MODEL_DIR). CI's "Retrieval quality gate" job requires it to run.
"""

import json
import os
from pathlib import Path

import pytest

from app.services.generation import topic_coverage
from app.services.rag.evidence import Passage
from tests.unit.test_topic_coverage_dataset import DATASET, evaluate

MODEL_DIR = os.getenv("EMBEDDING_MODEL_DIR")
REQUIRED = os.getenv("RETRIEVAL_QUALITY_REQUIRED") == "1"

ready = os.getenv("DATABASE_URL", "").startswith("postgresql") and bool(
    MODEL_DIR and (Path(MODEL_DIR) / "model.onnx").is_file()
)
if REQUIRED and not ready:
    raise RuntimeError("the topic-coverage gate needs DATABASE_URL and EMBEDDING_MODEL_DIR")
pytestmark = pytest.mark.skipif(not ready, reason="needs PostgreSQL and the model files")


async def test_topic_coverage_does_not_regress_below_the_baseline():
    from app.db.session import get_async_session_local, initialize_database, reset_engine
    from app.services.rag.onnx_embedder import OnnxMiniLMEmbedder
    from app.services.rag.pg_index import PgChunkIndex
    from app.services.rag.scopes import PgScopeStore

    baseline = json.loads((DATASET / "baseline.json").read_text(encoding="utf-8"))
    dataset = evaluate.load_dataset()
    await initialize_database()
    try:
        report = await evaluate.evaluate(
            dataset,
            PgChunkIndex(get_async_session_local()),
            OnnxMiniLMEmbedder(MODEL_DIR, threads=os.cpu_count() or 1),
            PgScopeStore(get_async_session_local()),
        )
    finally:
        await reset_engine()
    out = os.getenv("TOPIC_COVERAGE_REPORT")
    if out:
        Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    assert report["dataset"] == baseline["dataset"]
    assert report["coverage"]["selected"] == baseline["coverage"]["selected"]

    # The production check decides each topic as the evaluated signal did.
    passages = report["passages"]
    production = {
        tid
        for tid, t in dataset.topics.items()
        if not topic_coverage.covers(
            t["topic"],
            [Passage(n, text, f"c{n}", "d", 1, 1) for n, text in enumerate(passages[tid], 1)],
        )
    }
    assert production == set(report["verdicts"]["coverage_rejected"])

    regressions = []
    for part in ("coverage", "end_to_end"):
        for split in evaluate.SPLITS:
            now, base = _split(report, part, split), _split(baseline, part, split)
            for kind in ("false_positive", "false_negative"):
                if now[kind]["errors"] > base[kind]["errors"] + 1:
                    regressions.append(
                        f"{part} {split} {kind}: {now[kind]['errors']} > baseline "
                        f"{base[kind]['errors']}"
                    )
    assert not regressions, regressions


def _split(report: dict, part: str, split: str) -> dict:
    if part == "end_to_end":
        return report["end_to_end"][split]
    selected = report["coverage"]["selected"]["signal"]
    candidate = report["coverage"]["candidates"][selected]
    return candidate[split]
