"""The retrieval evaluation set (P3-5) is consistent and versioned.

Every evidence span is on its labelled page and on no distractor, the
committed distractors are exactly the generator's output, and the baseline
names this exact dataset: changing the set without re-baselining fails.
"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "retrieval_eval" / "v1"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluate = _script("evaluate_retrieval")
generator = _script("generate_retrieval_distractors")


def test_every_evidence_span_is_on_its_page_and_unambiguous():
    dataset = evaluate.load_dataset(DATASET)
    categories = {q["category"] for q in dataset.questions}
    assert {"direct", "paraphrase", "informal", "math", "multi_passage"} <= categories
    assert sum(q["category"] != evaluate.NO_ANSWER for q in dataset.questions) >= 50


def test_the_committed_distractors_are_the_generators_output():
    committed = json.loads((DATASET / "distractors.json").read_text(encoding="utf-8"))
    assert committed == generator.generate(committed["seed"])


def test_the_baseline_was_measured_on_this_exact_dataset():
    baseline = json.loads((DATASET / "baseline.json").read_text(encoding="utf-8"))
    assert baseline["dataset"]["sha256"] == evaluate.load_dataset(DATASET).sha256
    assert set(baseline["strategies"]) == set(evaluate.STRATEGIES)


def test_a_hit_needs_the_expected_span_not_just_the_document():
    question = {
        "expected": [{"document": "doc", "page": 2, "evidence": "the key sentence"}],
    }

    class Chunk:
        def __init__(self, document, page, content):
            self.document_id, self.content = document, content
            self.metadata = {"page_start": page, "page_end": page}

    same_doc_wrong_text = Chunk("doc", 2, "an unrelated sentence")
    right_text_wrong_page = Chunk("doc", 3, "the key sentence")
    right = Chunk("doc", 2, "before. The key sentence, after.".lower())
    assert evaluate.first_full_rank(question, [same_doc_wrong_text, right_text_wrong_page]) is None
    assert evaluate.first_full_rank(question, [same_doc_wrong_text, right]) == 2
