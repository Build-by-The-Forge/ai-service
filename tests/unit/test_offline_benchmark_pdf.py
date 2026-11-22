"""benchmarks/offline/pdf_eval.py: provenance, OCR flags and verified-only tasks."""

import importlib.util
import sys
from pathlib import Path

import fitz
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "offline"))
_spec = importlib.util.spec_from_file_location(
    "offline_pdf", ROOT / "benchmarks" / "offline" / "pdf_eval.py"
)
pdf_eval = importlib.util.module_from_spec(_spec)
sys.modules["offline_pdf"] = pdf_eval
_spec.loader.exec_module(pdf_eval)

_bench_spec = importlib.util.spec_from_file_location(
    "offline_bench_b", ROOT / "benchmarks" / "offline" / "bench.py"
)
bench = importlib.util.module_from_spec(_bench_spec)
_bench_spec.loader.exec_module(bench)


@pytest.fixture
def pdf_dir(tmp_path):
    folder = tmp_path / "pdfs"
    folder.mkdir()
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Ohm's law: the current through a resistor is V divided by R.")
    scanned = doc.new_page()
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), False)
    scanned.insert_image(fitz.Rect(0, 0, 100, 100), pixmap=pixmap)
    doc.save(folder / "notes.pdf")
    return folder


def test_extract_records_hash_pages_and_scanned_pages(pdf_dir):
    corpus = pdf_eval.extract(pdf_dir)
    doc = corpus["documents"][0]
    assert doc["doc_id"] == "D01" and len(doc["sha256"]) == 64
    assert [p["needs_ocr"] for p in doc["pages"]] == [False, True]
    assert pdf_eval.stats(corpus) == {
        "documents": 1,
        "pages": 2,
        "text_pages": 1,
        "pages_needing_ocr": 1,
        "empty_pages": 0,
    }
    # Scanned pages never become evidence until OCR has run.
    assert [p["id"] for p in pdf_eval.passages(corpus)] == ["D01-P1"]


def verified(dataset, **changes):
    task = dict(dataset["tasks"][0], review="verified", reviewer="A. Reviewer")
    task.update(question="What is the current through a resistor?", facts=[["V divided by R"]])
    task.update(changes)
    return dict(dataset, tasks=[task])


def test_template_slots_are_drafts_and_drafts_are_never_run(pdf_dir, tmp_path):
    dataset = pdf_eval.template(pdf_eval.extract(pdf_dir))
    assert [t["review"] for t in dataset["tasks"]] == ["draft"]
    path = tmp_path / "b.json"
    path.write_text(__import__("json").dumps(dataset))
    tasks, _ = bench.load_extra(path)
    assert tasks == []


def test_validate_accepts_a_checked_task_and_rejects_unsupported_facts(pdf_dir):
    dataset = pdf_eval.template(pdf_eval.extract(pdf_dir))
    assert pdf_eval.validate(verified(dataset)) == []
    wrong = pdf_eval.validate(verified(dataset, facts=[["I times R"]]))
    assert wrong and "not found on its supporting pages" in wrong[0]
    anonymous = pdf_eval.validate(verified(dataset, reviewer=""))
    assert any("named reviewer" in p for p in anonymous)
