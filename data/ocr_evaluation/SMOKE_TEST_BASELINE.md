# OCR Baseline: Smoke Test Phase

**Objective**: Prove that the actual PP-OCRv3/OpenVINO integration works end-to-end with minimal data before expanding to full corpus.

**Scope**: 3 documents only (2 images + 1 representative PDF)  
**Effort**: ~4-6 hours (ground truth) + ~30 min execution  
**Success Criteria**: Real OCR output → real CER/WER/latency → baseline report  

---

## Why Smoke Test First?

**The Risk We're Avoiding**:
- Spend 12-18 hours creating ground truth for 7 documents
- Run benchmark and discover the OCR runtime integration has issues
- Rework the entire pipeline
- Re-run ground truth collection

**The Smarter Approach**:
- Test with 3 documents only
- Prove the path works end-to-end
- Capture actual metrics (not estimates)
- Then expand confidently to full corpus

---

## Phase 0.1: Select Smoke-Test Corpus

### Documents Selected

| # | File | Type | Size | Usage |
|---|------|------|------|-------|
| 1 | `DTS SECOND SEMESTER EXAMINATION.jpeg` | Image | 129 KB | Full image |
| 2 | `IMG-20260627-WA0036.jpg` | Image | 118 KB | Full image |
| 3 | `MTS 102 Multiple Choice Limits.pdf` | PDF | 4.45 MB | **3 pages only** |

### Why This Selection

**Images**:
- 2 distinct document types (exam paper + mobile photo)
- Representative of BUILD's real intake
- Small enough for fast turnaround
- Tests image path completeness

**PDF (MTS 102 Multiple Choice)**:
- **Largest file** (4.45 MB) → likely scanned/image-based (needs OCR)
- **Academic content** (MCQs, equations) → tests real BUILD use case
- **18 total pages** → manageable to sample 3 pages
- **Moderate complexity** → not trivial, but not overwhelming

### Sampling Strategy for PDF

Sample 3 representative pages to test different structures:

| Page | Content Type | Why Chosen |
|------|--------------|-----------|
| 1 | Title + first MCQs | Tests heading + question parsing |
| 10 | Mid-document MCQs + possible equations | Tests mid-content consistency |
| 18 | Final page (possibly solutions/summary) | Tests end-of-document handling |

**Total PDF content for smoke test**: ~3-4 pages of text (out of 18)

---

## Phase 0.2: Minimal Ground Truth Annotation

### What to Annotate

**For each document**:
- Create JSON file in `data/ocr_evaluation/ground_truth/`
- Populate `ground_truth.full_text` with manually transcribed text
- Annotate **only the visible, readable content**
- Do NOT "correct" ambiguities

### Annotation Rules

1. **Type exactly what you see** — No cleanup
2. **Preserve structure** — Questions, options, spacing matter
3. **Mark illegible content** — `[illegible]` or `[unclear]`
4. **Mark non-text** — `[diagram]`, `[chart]`, `[image]`
5. **Superscripts/subscripts** — Use notation like `x^2`, `H_2O`
6. **Mathematical symbols** — `∫`, `∆`, `θ` (use Unicode if available)

### Example: Image Ground Truth

**Source**: `DTS SECOND SEMESTER EXAMINATION.jpeg`

```json
{
  "document": {
    "filename": "DTS SECOND SEMESTER 20252026 EXAMINATION.jpeg",
    "document_type": "examination_paper",
    "content_type": "raster_image",
    "page_count": 1,
    "quality_notes": "Camera-scanned exam paper"
  },
  "ground_truth": {
    "full_text": "DEPARTMENT OF EDUCATION\nDTS SECOND SEMESTER 2025/2026 EXAMINATION\n\nCourse: Introduction to Mathematics\nTime: 2 hours\n\nINSTRUCTIONS:\n1. Answer all questions\n2. Show all working\n\nSECTION A: Multiple Choice\n\n1. What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6\n\n2. Solve for x: x + 5 = 10\nA. 3\nB. 4\nC. 5\nD. 6\n\n[Continue with remaining questions...]"
  }
}
```

### Example: PDF Ground Truth (3 Pages Only)

**Source**: `MTS 102 Multiple Choice Limits.pdf` (pages 1, 10, 18)

```json
{
  "document": {
    "filename": "MTS 102  Multiple Choice Questions on Limits and Continuity BY ROTIMI.pdf",
    "document_type": "course_material",
    "content_type": "pdf_scanned",
    "page_count": 18,
    "quality_notes": "Scanned PDF, academic content"
  },
  "ground_truth": {
    "full_text": "MTS 102: MULTIPLE CHOICE QUESTIONS ON LIMITS AND CONTINUITY\nBy: Rotimi\n\n---PAGE 1---\n\n1. Find the limit as x approaches 2:\nlim (x→2) (x² + 3x - 1)\nA. 7\nB. 8\nC. 9\nD. 10\n\n2. Determine if the function is continuous at x = 0:\nf(x) = {x² if x ≠ 0, 1 if x = 0}\nA. Continuous\nB. Not continuous\nC. Discontinuity removable\nD. Discontinuity jump\n\n[Continue with page 1 content...]\n\n---PAGE 10---\n\n[Page 10 content...]\n\n---PAGE 18---\n\n[Page 18 content...final questions/solutions]"
  }
}
```

### Files to Create

1. ✅ `data/ocr_evaluation/ground_truth/DTS_SECOND_SEMESTER_EXAMINATION.json`
2. ✅ `data/ocr_evaluation/ground_truth/IMG-20260627-WA0036.json`
3. ✅ `data/ocr_evaluation/ground_truth/MTS_102_Multiple_Choice_Limits.json` **(3 pages only)**

**Estimated time**: 2-3 hours for images, 2-3 hours for PDF = 4-6 hours total

---

## Phase 0.3: Execute Real OCR Path

### Integration Point: Use Actual App OCR Service

**Do NOT bypass** the production OCR abstraction.

Instead, use the **existing OCR service** that BUILD's application would use:

```python
# Use the actual OCR provider from the app
from app.services.ocr.router import get_ocr_provider
from app.core.documents.intake import DocumentIntake

async def run_smoke_test():
    ocr_provider = get_ocr_provider()
    
    # Test image path
    image_path = Path("data/ocr_evaluation/images/DTS_SECOND_SEMESTER_EXAMINATION.jpeg")
    result = await ocr_provider.extract_text(image_path)
    print(result.extracted_text)
    
    # Test PDF path  
    pdf_path = Path("data/ocr_evaluation/pdfs/MTS 102...")
    result = await ocr_provider.extract_text(pdf_path)
    print(result.extracted_text)
```

**Why**: 
- Proves the production path works
- Captures real integration issues (if any)
- Results are representative of actual BUILD OCR behavior

### Expected Outcomes

For **each document/page**, capture:

| Metric | Type | Example |
|--------|------|---------|
| **OCR Success** | bool | `true` / `false` |
| **OCR Text** | string | Full extracted text |
| **CER** | float | `3.2%` |
| **WER** | float | `6.8%` |
| **Latency** | float | `1205 ms` |
| **Pages/sec** | float | `0.83` |
| **Memory Peak** | float | `512 MB` |
| **Errors** | list | `["font_error", "...]` |

**Example Result**:
```json
{
  "filename": "DTS_SECOND_SEMESTER_EXAMINATION.jpeg",
  "page": 1,
  "ocr_success": true,
  "ocr_text": "[extracted text from OCR]",
  "metrics": {
    "cer": 3.2,
    "wer": 6.8,
    "latency_ms": 850,
    "pages_per_second": 1.18,
    "peak_memory_mb": 481
  },
  "errors": []
}
```

---

## Phase 0.4: Capture Real Measurements

### Measurement Script

Create `scripts/smoke_test_benchmark.py`:

```python
#!/usr/bin/env python3
"""
Smoke test baseline: 3 documents, real PP-OCRv3 execution.
"""
import asyncio
import json
import time
from pathlib import Path
from typing import Any

async def run_smoke_test():
    """Execute OCR on 3 documents and capture metrics."""
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "candidate": "ppocr_v3",
        "runtime": "openvino",
        "documents": []
    }
    
    # Documents to test
    corpus = [
        ("data/ocr_evaluation/images/DTS SECOND SEMESTER EXAMINATION.jpeg", "image", 1),
        ("data/ocr_evaluation/images/IMG-20260627-WA0036.jpg", "image", 1),
        ("data/ocr_evaluation/pdfs/MTS 102 Multiple Choice.pdf", "pdf", [1, 10, 18]),
    ]
    
    for doc_path, doc_type, pages in corpus:
        doc_results = await process_document(
            doc_path, 
            doc_type, 
            pages
        )
        results["documents"].extend(doc_results)
    
    # Aggregate
    results["aggregate"] = aggregate_metrics(results["documents"])
    
    # Save
    output_file = Path("results/smoke_test_baseline.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

async def process_document(doc_path: str, doc_type: str, pages):
    """Process single document or pages."""
    # Real implementation would:
    # 1. Load document
    # 2. If PDF, extract specified pages
    # 3. Run OCR on each page
    # 4. Compare to ground truth
    # 5. Collect metrics
    pass

def aggregate_metrics(doc_results):
    """Aggregate CER/WER/latency across documents."""
    pass
```

### Capture Real Numbers

**Important**: Do not estimate, only record actual measurements.

```
✅ REAL: "latency_ms": 1205  (measured from actual run)
❌ FAKE: "latency_ms": 1000  (estimated/rounded)

✅ REAL: "cer": 3.2            (calculated from ground truth comparison)
❌ FAKE: "cer": 3.0            (assumed typical value)

✅ REAL: "pages_per_second": 0.83  (calculated from measured latency)
❌ FAKE: "pages_per_second": 1.0   (theoretical throughput)
```

---

## Phase 0.5: Qualitative Observations

After capturing metrics, **manually inspect OCR outputs** for:

### Critical for Educational Documents

| Aspect | Questions | Examples |
|--------|-----------|----------|
| **Equations** | Are mathematical symbols preserved? | x² vs x2, ∫ vs I |
| **MCQ Options** | Are question-option associations clear? | A. 15 B. 25 C. 35 D. 45 |
| **Headings** | Are titles/section breaks recognized? | Page breaks, question numbers |
| **Numbering** | Are numbered lists intact? | 1. 2. 3. vs 1 2 3 |
| **Superscripts** | Are exponents/subscripts preserved? | H₂O vs H2O, x^n vs xn |
| **Columns** | Is reading order correct? | Multi-column layouts |
| **Symbols** | Are special characters rendered? | θ, ∆, ℜ, ∑ |

### Observations Template

```markdown
## OCR Quality Assessment

### DTS SECOND SEMESTER EXAMINATION (Image)
- **Equations**: ✅ Preserved correctly (e.g., x² as x²)
- **MCQ Format**: ✅ Options correctly associated with questions
- **Headings**: ✅ Section breaks and titles recognized
- **Notes**: Overall quality is good; small OCR issues in question 15 (typo "quesiton")

### IMG-20260627-WA0036 (Image)
- **Perspective**: ⚠️ Mild perspective distortion detected; text still readable
- **Readability**: ✅ Most text extracted accurately
- **Issues**: "2025" extracted as "2025" (correct)

### MTS 102 Multiple Choice (PDF Pages 1, 10, 18)
- **Page 1 (Introduction)**: ✅ Title and early MCQs correct
- **Page 10 (Mid-document)**: ⚠️ Equation with fraction not preserved (a/b vs a|b)
- **Page 18 (Final page)**: ✅ Final questions extracted correctly
- **Overall**: Equations need attention; MCQ structure preserved well
```

---

## Phase 0.6: Baseline Report

After execution, produce a **concise report**:

**File**: `results/SMOKE_TEST_REPORT.md`

```markdown
# PP-OCRv3 Smoke Test Baseline Report

**Date**: 2026-09-01  
**Candidate**: PP-OCRv3 (OpenVINO)  
**Documents Tested**: 3 (2 images, 1 PDF with 3 sampled pages)  
**Execution Status**: ✅ Success (all documents processed)

## Quantitative Results

| Metric | Value |
|--------|-------|
| **Avg CER** | 3.8% |
| **Avg WER** | 7.6% |
| **Mean Latency** | 950 ms/page |
| **Throughput** | 1.05 pages/sec |
| **Peak Memory** | 512 MB |

### Per-Document Breakdown

#### Images
- **DTS Exam**: CER 3.2%, WER 6.8%, Latency 850ms
- **Mobile Photo**: CER 5.1%, WER 9.2%, Latency 920ms

#### PDF (3 Pages)
- **Page 1**: CER 3.1%, WER 6.2%, Latency 1020ms
- **Page 10**: CER 4.5%, WER 8.9%, Latency 980ms
- **Page 18**: CER 2.8%, WER 5.4%, Latency 900ms

## Qualitative Assessment

### Strengths
- ✅ MCQ format preserved correctly
- ✅ Section headings recognized
- ✅ Exam structure maintained

### Issues
- ⚠️ Mathematical notation (fractions, subscripts) occasionally misread
- ⚠️ Mobile photo shows some perspective-induced errors
- ⚠️ Dense formatting in cheat sheet section needs review

## Recommendations

### Baseline Assessment
- **Status**: ✅ ACCEPTABLE
- **CER/WER**: Within reasonable range for educational documents
- **Path Integrity**: ✅ Confirmed (image + PDF paths work end-to-end)

### Next Steps
1. ✅ Smoke test passed — proceed to expand corpus
2. ⏳ Acquire additional documents (target: 50+ samples)
3. ⏳ Test newer PaddleOCR version
4. ⏳ Benchmark Surya model
5. ⏳ Compare results and select production candidate

## Failure Cases
- **None in smoke test** (all 3 documents processed successfully)

## Deployment Readiness
- **Integration**: ✅ Production OCR path confirmed working
- **Performance**: ✅ Acceptable latency for batch processing
- **Resource Usage**: ✅ Memory usage reasonable
- **Educational Content**: ⚠️ Equation handling could be improved (defer to model selection phase)
```

---

## Execution Checklist

### Ground Truth Population (4-6 hours)

- [ ] Examine `DTS SECOND SEMESTER EXAMINATION.jpeg` visually
- [ ] Transcribe all visible text to `DTS_SECOND_SEMESTER_EXAMINATION.json`
- [ ] Validate JSON is parseable: `python -m json.tool file.json`

- [ ] Examine `IMG-20260627-WA0036.jpg` visually
- [ ] Transcribe all visible text to `IMG-20260627-WA0036.json`
- [ ] Validate JSON

- [ ] Extract/render pages 1, 10, 18 from `MTS 102 Multiple Choice.pdf`
- [ ] Transcribe text from those 3 pages only to `MTS_102_Multiple_Choice_Limits.json`
- [ ] Validate JSON

### OCR Execution (30 min - 1 hour)

- [ ] Ensure OpenVINO + PaddleOCR installed: `pip install -r requirements.txt`
- [ ] Verify model files present in `models/ocr/openvino_models/`
- [ ] Run smoke test benchmark:
  ```bash
  python scripts/smoke_test_benchmark.py --corpus-dir data/ocr_evaluation
  ```
- [ ] Monitor for errors; capture actual OCR text output
- [ ] Verify output saved to `results/smoke_test_baseline.json`

### Analysis & Reporting (1-2 hours)

- [ ] Load `results/smoke_test_baseline.json`
- [ ] Verify metrics are real (not estimated)
- [ ] Inspect OCR outputs for qualitative issues (equations, MCQs, layout)
- [ ] Create `results/SMOKE_TEST_REPORT.md` with findings
- [ ] Document any runtime issues encountered

### Decision Point

**If smoke test PASSES** (all documents process, metrics captured):
- ✅ Proceed to corpus expansion phase (50+ documents)
- ✅ Acquire newer models if needed
- ✅ Prepare for multi-candidate benchmarking

**If smoke test FAILS** (runtime error, 0% success, OOM, etc.):
- ❌ Diagnose the minimum issue (integration, model path, memory, etc.)
- ❌ Fix only that issue
- ❌ Re-run smoke test
- ❌ Do NOT expand scope to new models yet

---

## Success Criteria (Smoke Test)

- ✅ All 3 documents process without unrecoverable errors
- ✅ OCR text output captured (not empty)
- ✅ CER/WER/latency metrics calculated from real measurements
- ✅ Metrics are reproducible across multiple runs
- ✅ Qualitative assessment completed
- ✅ No fabricated numbers in report
- ✅ Integration with app OCR service verified

---

## Files to Create/Use

| File | Purpose | Status |
|------|---------|--------|
| `data/ocr_evaluation/ground_truth/DTS_SECOND_SEMESTER_EXAMINATION.json` | Image 1 GT | ⏳ Create |
| `data/ocr_evaluation/ground_truth/IMG-20260627-WA0036.json` | Image 2 GT | ⏳ Create |
| `data/ocr_evaluation/ground_truth/MTS_102_Multiple_Choice_Limits.json` | PDF (3 pages) GT | ⏳ Create |
| `scripts/smoke_test_benchmark.py` | Benchmark execution script | ⏳ Create |
| `results/smoke_test_baseline.json` | Raw benchmark results | ⏳ Generated |
| `results/SMOKE_TEST_REPORT.md` | Analysis and findings | ⏳ Create |

---

## Timeline

| Phase | Task | Time |
|-------|------|------|
| 0.2 | Ground truth (3 docs) | 4-6 hours |
| 0.3-0.4 | OCR execution + metrics | 30 min - 1 hour |
| 0.5-0.6 | Analysis + report | 1-2 hours |
| **Total** | | **6-9 hours** |

---

## Key Principle

> **No fabricated metrics. Every number in the report must come from actual measured OCR execution.**

If we get 3.8% CER, it's because the actual ground truth comparison produced that result.  
If latency is 950ms, it's because that's what the clock measured.  
If we can't measure it, we don't report it.

---

**Status**: Ready to start Phase 0.2 (ground truth annotation)  
**Next Action**: Transcribe the 2 images + 3 pages of PDF  
**Owner**: BUILD AI/ML Team  
**Created**: 2026-09-01
