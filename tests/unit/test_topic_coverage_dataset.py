"""The topic-coverage evaluation set (data/topic_coverage/v1) is consistent.

Every covered topic names a verbatim span on its page; every uncovered one
says why; a labelled pair never mixes the dev and held-out test splits; and
every question is one the existing support and exam checks keep, so each
off-topic pair is the hard case: supported by its passage, but not about
the topic.
"""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

from app.schemas.ai import GeneratedQuestion
from app.services.generation.cbt_exam import check_exam_question
from app.services.generation.question_checks import check_question
from app.services.rag.evidence import Evidence, Passage
from tests.rag_support import sources_for

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "topic_coverage" / "v1"


def _script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluate = _script("evaluate_topic_coverage")


@pytest.fixture(scope="module")
def dataset():
    return evaluate.load_dataset()


def test_the_set_has_hard_near_misses_and_out_of_corpus_topics_in_both_splits(dataset):
    counts = Counter((t["split"], t["category"]) for t in dataset.topics.values())
    for split in evaluate.SPLITS:
        assert counts[(split, "near_miss")] >= 30
        assert counts[(split, "out_of_corpus")] >= 15
        covered = sum(n for (s, c), n in counts.items() if s == split and c.startswith("covered"))
        assert covered >= 40
        for category in ("covered_direct", "covered_paraphrase", "covered_broad"):
            assert counts[(split, category)] >= 5


def test_every_near_miss_is_in_a_course_the_corpus_teaches(dataset):
    taught = {t["course"] for t in dataset.topics.values() if t["covered"]}
    near = {t["course"] for t in dataset.topics.values() if t["category"] == "near_miss"}
    other = {t["course"] for t in dataset.topics.values() if t["category"] == "out_of_corpus"}
    assert near <= taught
    assert not other & taught


def test_every_question_is_kept_by_the_support_and_exam_checks(dataset):
    kept = []
    for q in dataset.questions:
        passages = tuple(
            Passage(n, dataset.pages[p["document"]][p["page"] - 1], f"c{n}", p["document"], 1, 1)
            for n, p in enumerate(q["passages"], start=1)
        )
        evidence = Evidence("", passages, sources_for(*passages))
        support = check_question(q["item"], evidence)
        exam = check_exam_question(GeneratedQuestion.model_validate(q["item"]), passages)
        kept.append((q["id"], support.reason, exam.reason))
    assert [k for k in kept if k[1:] != (None, None)] == []


def test_each_question_has_hard_and_easy_off_topic_pairs(dataset):
    for q in dataset.questions:
        categories = {dataset.topics[t]["category"] for t in q["off_topic"]}
        assert "near_miss" in categories and "out_of_corpus" in categories, q["id"]


def test_a_mixed_split_pair_is_refused(tmp_path):
    topics = (DATASET / "topics.json").read_text(encoding="utf-8")
    questions = (DATASET / "questions.json").read_text(encoding="utf-8")
    # q-mts-01 is a dev question; mts-c02 is a test topic.
    (tmp_path / "topics.json").write_text(topics, encoding="utf-8")
    (tmp_path / "questions.json").write_text(
        questions.replace('"mts-c03",', '"mts-c02",', 1), encoding="utf-8"
    )
    with pytest.raises(evaluate.DatasetError, match="another split"):
        evaluate.load_dataset(tmp_path)


def test_the_dataset_identity_does_not_depend_on_line_endings(tmp_path, dataset):
    for name in ("topics.json", "questions.json"):
        lf = (DATASET / name).read_bytes().replace(b"\r\n", b"\n")
        (tmp_path / name).write_bytes(lf.replace(b"\n", b"\r\n"))
    assert evaluate.load_dataset(tmp_path).sha256 == dataset.sha256
