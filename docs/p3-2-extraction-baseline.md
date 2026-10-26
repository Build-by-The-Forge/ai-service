# P3-2 extraction baseline (before any pipeline change)

Measured on 2026-10-26 with `scripts/evaluate_ocr_ground_truth.py`. It ran in the AI-service image (Pillow 13, paddle 3.3.1, verified PP-OCRv3 models) with networking disabled and the corpus read-only, in 649 s. Only metrics are recorded here, never recognised text.

## Ground truth

`data/ocr_evaluation/ground_truth/*.json` named nine pages across the seven evaluation documents, but held placeholders, not transcriptions. They now hold **draft** transcriptions (`ground_truth_type: draft_pending_human_verification`) made from the rendered pages:

- reading order;
- mathematics written linearly with Unicode symbols;
- diagonal watermarks over the content omitted.

**These figures become the official baseline only after a person verifies the drafts.** Then they are re-measured, and this document is updated.

## Results

CER is character error rate and WER is word error rate. Both are edit distance divided by the reference length, after Unicode NFC and whitespace normalization (`app/services/ingestion/text_metrics.py`). "OCR" is the current pipeline: PaddleOCR on the page rendered at 2x. "Text layer" is the PDF's embedded text (PyMuPDF), which P3-2 will use for native pages.

| Document, page | Kind | OCR CER | OCR WER | Text-layer CER |
|---|---|---|---|---|
| DTS 104 examination (photo) | printed, photographed | 0.021 | 0.192 | — |
| Handwritten lab notes (photo) | handwriting | 0.581 | 0.867 | — |
| MTS102 Solutions, p1 | LaTeX PDF | 0.154 | 0.269 | **0.026** |
| MTS 102 Cheat Sheet, p1 | PDF with watermark | **0.035** | 0.153 | 0.704 |
| Differentiation worksheet, p1 | LaTeX PDF, dense maths | 0.362 | 0.771 | **0.128** |
| Differentiation worksheet, p2 | LaTeX PDF, dense maths | 0.654 | 0.919 | **0.213** |
| Limits MCQ, p1 | scanned, maths | 0.407 | 0.602 | no text layer |
| PHY 102 MCQ, p1 | PDF, prose | **0.020** | 0.040 | 0.071 |
| **All OCR (8 pages, 10,263 reference chars)** | | **0.260** | **0.456** | |
| **All text layer (6 PDF pages, 7,750 chars)** | | | | **0.236** (WER 0.475) |

## What it says for P3-2

1. **A good text layer beats OCR by a wide margin** on digitally produced PDFs, especially mathematics: 2.6% against 15.4%, and 13–21% against 36–65%.
2. **A text layer's reading order must be decided per page.** The cheat sheet's layer first scored 70% CER, but its text is correct: it is stored in drawing order (sections come out 1, 3, 2) and uses typographic ligatures (`ﬀ`, `ﬁ`). Two extraction orders over the same text layer give:

   | Page | Drawing order | Position-sorted, ligatures expanded |
   |---|---|---|
   | Cheat sheet | 0.703 | **0.010** |
   | PHY 102 | 0.071 | **0.001** |
   | MTS102 Solutions (LaTeX) | **0.026** | 0.077 |
   | Differentiation p1 / p2 (LaTeX) | **0.128 / 0.213** | 0.256 / 0.451 |

   - LaTeX output is already in reading order, and sorting fragments its mathematics.
   - Office-style exports draw in arbitrary order, and sorting fixes it.

   P3-2 therefore picks the order per page (for example, by how often the drawing order jumps upward), always expands ligatures, and falls back to OCR only when no usable layer exists. With the better order per page, the text layer scores 0.001–0.213 where it exists, against OCR's 0.020–0.654.
3. **Scanned mathematics is the weakest case** (41% on the Limits sheet). It is the largest opportunity, and the one this OCR model handles worst.
4. **Handwriting (58%) is out of reach** for this printed-text model. It is reported per page so it never hides behind an average.

## Gate

P3-2 changes are measured with the same script on the same pages. The no-regression rule:

- the overall CER may not rise;
- no page's CER may rise by more than 0.02 without an explanation recorded here.

Pages where P3-2 changes the method (text layer instead of OCR) are reported for both methods.

## The broader corpus (aggregate only)

`data/ocr_evaluation/pdfs/1st Semester/` holds real first-semester course material: BIO 101/103, CHE 101, CVE 105 and more. That is 175 files (141 PDFs, plus .doc/.docx/.ppt/.pptx and images). It is kept out of git, at 292 MB and pending a licence review. Surveyed with PyMuPDF (no text read out):

| | Pages | Share |
|---|---|---|
| Native text (≥ 200 characters, images under half the page) | 1,725 | 53.1% |
| Scanned or image (images cover half the page or more) | 1,390 | 42.8% |
| Sparse text | 68 | 2.1% |
| Mixed (text and large images) | 36 | 1.1% |
| Empty | 31 | 1.0% |
| **Total** (141 documents, median 12 pages, largest 538) | **3,250** | |

None were encrypted or unreadable. For P3-2:

- **Over half of all pages need no OCR.** A text layer is read in milliseconds, against seconds of CPU OCR per page, so native classification is the largest speed gain available.
- **Page and time limits are required.** One upload has 538 pages.
- **The non-PDF formats (.doc, .docx, .ppt, .pptx) are not accepted by upload today** (PDF, PNG and JPEG only). Supporting them is a separate decision, not part of P3-2.
