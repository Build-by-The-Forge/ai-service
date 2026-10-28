"""Build a local retrieval-evaluation corpus from a folder of PDFs (P3-5 tier 2).

    python scripts/build_retrieval_corpus.py PDF_DIR OUT_DIR

Writes OUT_DIR/corpus.json in the format scripts/evaluate_retrieval.py reads
(documents as lists of pages), plus an empty distractors.json. Each page is
read the way the P3-2 extractor reads it without OCR: native and mixed
pages through text_layer, with its per-page reading order. Pages that
would need OCR (scanned) and empty pages are kept as empty strings, so page numbers stay
true, and are counted in OUT_DIR/build.json.

The output holds document text. It is for local evaluation only: write it
outside the repository and never commit it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fitz  # noqa: E402

from app.services.ingestion.extraction import (  # noqa: E402
    MIXED,
    NATIVE,
    classify_page,
    text_layer,
)

TEXT_LAYER_KINDS = (NATIVE, MIXED)


def build(pdf_dir: Path, out_dir: Path) -> dict:
    documents = []
    kinds: Counter[str] = Counter()
    unreadable = 0
    for path in sorted(pdf_dir.rglob("*")):
        if path.suffix.lower() != ".pdf":
            continue
        try:
            pdf = fitz.open(path)
        except Exception:
            unreadable += 1
            continue
        with pdf:
            pages = []
            for page in pdf:
                kind = classify_page(page)
                kinds[kind] += 1
                pages.append(text_layer(page) if kind in TEXT_LAYER_KINDS else "")
        # An opaque id: the file name never leaves this machine via a report.
        doc_id = hashlib.sha256(str(path.relative_to(pdf_dir)).encode()).hexdigest()[:12]
        documents.append({"id": doc_id, "source": str(path.relative_to(pdf_dir)), "pages": pages})
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = {"version": "tier2-local", "documents": documents}
    (out_dir / "corpus.json").write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    (out_dir / "distractors.json").write_text(
        json.dumps({"version": "tier2-local", "documents": []}), encoding="utf-8"
    )
    summary = {
        "documents": len(documents),
        "unreadable": unreadable,
        "pages": sum(kinds.values()),
        "pages_by_kind": dict(sorted(kinds.items())),
        "text_layer_pages": sum(kinds[k] for k in TEXT_LAYER_KINDS),
    }
    (out_dir / "build.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.pdf_dir, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
