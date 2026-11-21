"""Dataset B: evaluation over real course PDFs, kept on the local machine.

The PDFs are third-party academic material. Nothing this script writes is
committed (benchmarks/offline/local/ is git-ignored), nothing is sent to any
external service, and the originals are opened read-only. Only aggregate
numbers (page counts, OCR needs, scores) may go into a report.

    # 1. Extract page text with provenance, flag scanned pages, hash every file.
    python benchmarks/offline/pdf_eval.py extract data/ocr_evaluation/pdfs \
        --out benchmarks/offline/local/corpus.json
    # 2. Make a review template: one empty task slot per text page.
    python benchmarks/offline/pdf_eval.py template benchmarks/offline/local/corpus.json \
        --out benchmarks/offline/local/dataset_b.json
    # 3. A person writes each question, answer facts and supporting pages, and sets
    #    "review": "verified" only after checking them against the PDF itself.
    # 4. Check the verified tasks against the extracted text, then run them:
    python benchmarks/offline/pdf_eval.py validate benchmarks/offline/local/dataset_b.json
    python benchmarks/offline/bench.py ... --extra-tasks benchmarks/offline/local/dataset_b.json

Generated questions are never treated as ground truth: a model may draft a slot,
but only a human-verified task is scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scoring  # noqa: E402

# A page with fewer extractable characters than this, and at least one image, is
# treated as scanned: its text needs OCR before it can be used as evidence.
SCANNED_CHARS = 80
PASSAGE_CHARS = 1800


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(pdf_dir: Path) -> dict[str, Any]:
    import fitz  # PyMuPDF, already an ai-service dependency

    documents = []
    for index, path in enumerate(sorted(pdf_dir.glob("*.pdf")), start=1):
        doc_id = f"D{index:02d}"
        pages = []
        with fitz.open(path) as pdf:
            for number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                images = len(page.get_images(full=False))
                pages.append(
                    {
                        "page": number,
                        "chars": len(text),
                        "images": images,
                        "needs_ocr": len(text) < SCANNED_CHARS and images > 0,
                        "text": text,
                    }
                )
        documents.append(
            {
                "doc_id": doc_id,
                "file": path.name,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "pages": pages,
            }
        )
    return {"schema": 1, "extractor": f"PyMuPDF {fitz.VersionBind}", "documents": documents}


def stats(corpus: dict[str, Any]) -> dict[str, Any]:
    pages = [p for d in corpus["documents"] for p in d["pages"]]
    return {
        "documents": len(corpus["documents"]),
        "pages": len(pages),
        "text_pages": sum(1 for p in pages if not p["needs_ocr"] and p["chars"] > 0),
        "pages_needing_ocr": sum(1 for p in pages if p["needs_ocr"]),
        "empty_pages": sum(1 for p in pages if p["chars"] == 0 and not p["needs_ocr"]),
    }


def passages(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    """One passage per usable page, id <doc>-p<page>, text trimmed to a phone-sized chunk."""
    out = []
    for doc in corpus["documents"]:
        for page in doc["pages"]:
            if page["needs_ocr"] or not page["chars"]:
                continue
            out.append(
                {
                    "id": f"{doc['doc_id']}-P{page['page']}",
                    "course": doc["doc_id"],
                    "title": f"page {page['page']}",
                    "text": page["text"][:PASSAGE_CHARS],
                    "source_sha256": doc["sha256"],
                }
            )
    return out


def template(corpus: dict[str, Any]) -> dict[str, Any]:
    items = passages(corpus)
    tasks = [
        {
            "id": f"B-{p['id']}",
            "type": "grounded",
            "lang": "en",
            "review": "draft",
            "passages": [p["id"]],
            "question": "",
            "answerable": True,
            "facts": [],
            "supporting": [p["id"]],
            "reviewer": "",
            "notes": "Write the question from this page; add distractor pages to 'passages'.",
        }
        for p in items
    ]
    return {"schema": 1, "passages": items, "tasks": tasks}


def validate(dataset: dict[str, Any]) -> list[str]:
    """Problems with verified tasks; drafts are ignored."""
    problems = []
    texts = {p["id"]: p["text"] for p in dataset["passages"]}
    for task in dataset["tasks"]:
        if task.get("review") != "verified":
            continue
        tid = task["id"]
        if not task.get("question"):
            problems.append(f"{tid}: no question")
        if not task.get("reviewer"):
            problems.append(f"{tid}: verified without a named reviewer")
        unknown = set(task["passages"]) - set(texts)
        if unknown:
            problems.append(f"{tid}: unknown passages {sorted(unknown)}")
        if not set(task["supporting"]) <= set(task["passages"]):
            problems.append(f"{tid}: supporting pages not among the passages")
        if task["answerable"]:
            if not task["facts"] or not task["supporting"]:
                problems.append(f"{tid}: answerable but no facts or supporting pages")
            evidence = " ".join(texts.get(p, "") for p in task["supporting"])
            for group in task["facts"]:
                if not any(scoring.matches(evidence, alt) for alt in group):
                    problems.append(f"{tid}: fact {group} not found on its supporting pages")
        elif task["facts"] or task["supporting"]:
            problems.append(f"{tid}: unanswerable tasks take no facts or supporting pages")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_extract = sub.add_parser("extract")
    p_extract.add_argument("pdf_dir", type=Path)
    p_extract.add_argument("--out", type=Path, required=True)
    p_template = sub.add_parser("template")
    p_template.add_argument("corpus", type=Path)
    p_template.add_argument("--out", type=Path, required=True)
    p_validate = sub.add_parser("validate")
    p_validate.add_argument("dataset", type=Path)
    args = parser.parse_args()

    if args.command == "extract":
        corpus = extract(args.pdf_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(stats(corpus), indent=1))  # aggregate only: safe to share
    elif args.command == "template":
        corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
        if args.out.exists():
            raise SystemExit(f"{args.out} exists; refusing to overwrite reviewed work")
        args.out.write_text(
            json.dumps(template(corpus), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"{len(passages(corpus))} page slots written to {args.out}")
    else:
        dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
        problems = validate(dataset)
        verified = sum(1 for t in dataset["tasks"] if t.get("review") == "verified")
        print(f"{verified} verified tasks, {len(problems)} problems")
        for problem in problems:
            print(" -", problem)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
