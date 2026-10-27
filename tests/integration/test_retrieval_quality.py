"""The retrieval quality gate (P3-5): recall@5 and MRR@10 must not regress
below the committed baseline (data/retrieval_eval/v1/baseline.json).

The evaluation runs the production path (chunker, local embedder,
PgChunkIndex) for each strategy. The baseline is a measured reference, not
a release target; the tolerance absorbs floating-point differences between
machines (one question of recall). Improving retrieval means re-measuring
and committing a new baseline (scripts/evaluate_retrieval.py
--write-baseline) with the change.

Needs PostgreSQL with pgvector (DATABASE_URL) and the verified model files
(EMBEDDING_MODEL_DIR). CI's "Retrieval quality gate" job requires it to run.
"""

import json
import os
from pathlib import Path

import pytest

from tests.unit.test_retrieval_eval_dataset import DATASET, evaluate

MODEL_DIR = os.getenv("EMBEDDING_MODEL_DIR")
REQUIRED = os.getenv("RETRIEVAL_QUALITY_REQUIRED") == "1"
MRR_TOLERANCE = 0.02

ready = os.getenv("DATABASE_URL", "").startswith("postgresql") and bool(
    MODEL_DIR and (Path(MODEL_DIR) / "model.onnx").is_file()
)
if REQUIRED and not ready:
    raise RuntimeError("the retrieval quality gate needs DATABASE_URL and EMBEDDING_MODEL_DIR")
pytestmark = pytest.mark.skipif(not ready, reason="needs PostgreSQL and the model files")


async def test_retrieval_quality_does_not_regress_below_the_baseline():
    from app.db.session import get_async_session_local, initialize_database, reset_engine
    from app.services.rag.onnx_embedder import OnnxMiniLMEmbedder
    from app.services.rag.pg_index import PgChunkIndex

    baseline = json.loads((DATASET / "baseline.json").read_text(encoding="utf-8"))
    await initialize_database()
    try:
        report = await evaluate.evaluate(
            evaluate.load_dataset(DATASET),
            PgChunkIndex(get_async_session_local()),
            OnnxMiniLMEmbedder(MODEL_DIR, threads=os.cpu_count() or 1),
        )
    finally:
        await reset_engine()
    out = os.getenv("RETRIEVAL_QUALITY_REPORT")
    if out:
        Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    assert report["dataset"] == baseline["dataset"]
    # One question of recall, a small margin of MRR.
    tolerance = {"recall@5": 1 / baseline["questions"] + 1e-9, "mrr@10": MRR_TOLERANCE}
    regressions = [
        f"{strategy} {metric}: {report['strategies'][strategy][metric]} < baseline "
        f"{base[metric]} (misses {report['strategies'][strategy]['misses']})"
        for strategy, base in baseline["strategies"].items()
        for metric, margin in tolerance.items()
        if report["strategies"][strategy][metric] < base[metric] - margin
    ]
    assert not regressions, regressions
