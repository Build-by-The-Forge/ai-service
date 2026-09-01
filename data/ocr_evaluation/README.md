# OCR Model Evaluation Dataset

**Phase:** AI-2B.4.3 — OCR Candidate Benchmark

This directory contains the shared evaluation corpus for benchmarking OCR candidates:

- **PP-OCRv3/OpenVINO** (baseline, from IDQORE)
- **Newer PaddleOCR** (primary candidate)
- **Surya** (accuracy-oriented candidate)

## Dataset Structure

### `images/`

Image-based OCR test cases (no preprocessing, direct OCR):

- `id_card.png` — standard identity document
- `passport.jpg` — travel document
- `certificate.jpg` — academic/credential document
- `academic_transcript.jpg` — multi-column grade record
- `lecture_notes.png` — handwritten/typed student notes
- `form_with_tables.png` — form containing tables/structured data
- `mathematical_text.jpg` — equations, scientific notation
- `low_quality.png` — compressed/noisy image
- `rotated_page.png` — 45-degree rotation
- `skewed_page.png` — perspective distortion

### `pdfs/`

PDF-based OCR test cases (may contain native text or scanned pages):

- `academic_transcript.pdf` — multi-page, native text
- `scanned_document.pdf` — fully scanned, no native text
- `mixed_document.pdf` — blend of native + scanned pages
- `form_with_tables.pdf` — structured data, table extraction
- `multi_column_notes.pdf` — complex layout
- `rotated_pages.pdf` — pages with various orientations

### `ground_truth/`

JSON files with expected OCR output and annotations:

```json
{
  "filename": "id_card.png",
  "document_type": "identity_document",
  "regions": [
    {
      "text": "John Doe",
      "bbox": [0.1, 0.15, 0.9, 0.25],
      "confidence_expected": 0.95,
      "field": "name"
    },
    {
      "text": "123-45-6789",
      "bbox": [0.1, 0.35, 0.9, 0.45],
      "confidence_expected": 0.90,
      "field": "id_number"
    }
  ],
  "metadata": {
    "languages": ["en"],
    "content_types": ["text"],
    "layout_complexity": "simple",
    "image_quality": "high"
  }
}
```

## Evaluation Metrics

For each candidate, we measure:

| Metric | Rationale |
|--------|-----------|
| **Text Accuracy** | Primary: word error rate, character error rate vs ground truth |
| **Detection Quality** | Bounding box precision and recall |
| **Layout Preservation** | Relative positioning of blocks, reading order |
| **Table Extraction** | Structured data accuracy |
| **Math/Scientific Text** | Special notation, symbols, equations |
| **Rotation/Skew Handling** | Angle tolerance and correction |
| **Speed (Latency)** | Time per page on CPU |
| **Memory Footprint** | Peak RAM during inference |
| **Language Coverage** | Multilingual support (English, if others relevant) |
| **Licensing** | Production suitability |
| **Deployment Fit** | OpenVINO compatibility, model size |

## Usage

### 1. Run all candidates against the corpus

```bash
# Baseline: PP-OCRv3
python scripts/benchmark_ocr.py --candidate ppocr_v3 --output results/ppocr_v3.json

# Candidate 1: Newer PaddleOCR
python scripts/benchmark_ocr.py --candidate paddleocr_new --output results/paddleocr_new.json

# Candidate 2: Surya
python scripts/benchmark_ocr.py --candidate surya --output results/surya.json
```

### 2. Compare results

```bash
python scripts/compare_ocr_results.py --results results/ --output comparison.html
```

### 3. Make selection

Based on evidence, select the production OCR stack and document the decision in `DECISION.md`.

## Important Notes

- **Do NOT edit ground truth** during evaluation
- **Same corpus for all candidates** — no special preprocessing per candidate
- **Document real-world BUILD use cases** — id cards, transcripts, lecture materials
- **Test both image and PDF paths** — OCR is not just images
- **Distinguish extraction vs OCR** — PDFs may have native text that doesn't require OCR

---

See `AI-2B.4.3 — OCR Model Evaluation` in milestone tracking for full context.
