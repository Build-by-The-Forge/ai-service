"""Extraction accuracy on the evaluation corpus's ground-truth pages (P3-2).

For every page named in data/ocr_evaluation/ground_truth/*.json, measures
CER and WER of:
- "ocr": real OCR (PaddleOCRService) of the page rendered at 2x, the
  current pipeline's input;
- "native": the PDF's embedded text layer (PyMuPDF), what P3-2 will use
  for native pages (PDFs only).

Run inside the AI-service image with networking disabled; prints JSON with
per-page figures and reference-length-weighted totals. Only metrics are
printed, never recognised text.

    python scripts/evaluate_ocr_ground_truth.py data/ocr_evaluation [--out baseline.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # noqa: E402

from app.services.ingestion.text_metrics import cer, normalize, wer  # noqa: E402
from app.services.ocr.paddle import PaddleOCRService  # noqa: E402

ZOOM = fitz.Matrix(2, 2)


def _source(corpus: str, filename: str) -> str:
    for sub in ("pdfs", "images"):
        path = os.path.join(corpus, sub, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(filename)


async def _ocr(service: PaddleOCRService, image_path: str) -> str:
    lines = await service.extract_text_from_image(image_path)
    return "\n".join(line["text"] for line in lines)


def _score(reference: str, hypothesis: str) -> dict:
    return {
        "cer": round(cer(reference, hypothesis), 4),
        "wer": round(wer(reference, hypothesis), 4),
        "reference_chars": len(normalize(reference)),
        "hypothesis_chars": len(normalize(hypothesis)),
    }


async def evaluate(corpus: str) -> dict:
    service = PaddleOCRService()
    pages: list[dict] = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        for name in sorted(os.listdir(os.path.join(corpus, "ground_truth"))):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(corpus, "ground_truth", name), encoding="utf-8") as f:
                data = json.load(f)
            document = name.removesuffix(".json")
            source = _source(corpus, data["document"]["filename"])
            truth = data["ground_truth"]
            status = data["metadata"].get("ground_truth_type")
            if source.lower().endswith(".pdf"):
                with fitz.open(source) as pdf:
                    for entry in truth["pages"]:
                        page = pdf[entry["page_number"] - 1]
                        image = os.path.join(tmp, f"{document}-{entry['page_number']}.png")
                        page.get_pixmap(matrix=ZOOM).save(image)
                        reference = entry["full_text"]
                        record = {
                            "document": document,
                            "page": entry["page_number"],
                            "status": status,
                        }
                        record["ocr"] = _score(reference, await _ocr(service, image))
                        record["native"] = _score(reference, page.get_text("text"))
                        pages.append(record)
            else:
                record = {"document": document, "page": 1, "status": status}
                record["ocr"] = _score(truth["full_text"], await _ocr(service, source))
                pages.append(record)

    def total(method: str) -> dict:
        scored = [p[method] for p in pages if method in p]
        chars = sum(s["reference_chars"] for s in scored)

        def weighted(key: str) -> float:
            return round(sum(s[key] * s["reference_chars"] for s in scored) / max(chars, 1), 4)

        return {
            "pages": len(scored),
            "reference_chars": chars,
            "cer": weighted("cer"),
            "wer": weighted("wer"),
        }

    return {
        "pages": pages,
        "totals": {"ocr": total("ocr"), "native": total("native")},
        "seconds": round(time.monotonic() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = asyncio.run(evaluate(args.corpus))
    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
