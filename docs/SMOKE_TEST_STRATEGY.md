# OCR Baseline Benchmarking: Strategic Shift to Smoke Test First

**Date**: 2026-09-01  
**Status**: ⏳ Ready for Ground Truth Annotation Phase  
**Decision Point**: After smoke test succeeds, proceed to corpus expansion

---

## Executive Summary

### Previous Plan ❌
- Create ground truth for all **7 documents** (12-18 hours of transcription)
- Run OCR on all 7 documents
- Analyze results
- **RISK**: Discover integration issue after all that work

### New Plan ✅
- Create ground truth for **3 documents only** (4-6 hours of annotation)
- Run OCR on 3 documents using **actual app service** (proven path)
- Analyze results and establish baseline
- **If successful** → expand to 50+ documents
- **If failed** → fix minimum issue and retry

**Net Impact**: Shorter feedback loop, proven integration before major investment

---

## Why This Approach

### The Problem with Full Corpus First

```
Timeline Cost:
Day 1-3:    Transcribe all 7 docs               (12-18 hours)
Day 4:      Discover OCR integration broken     (1 hour)
Day 5-7:    Fix integration                     (4-8 hours)
Day 8-10:   Re-transcribe docs (validation)     (6-12 hours)
            Total: 23-38 hours 😞

Risk: All ground truth work invalidated by integration issues
```

### The Smart Approach

```
Timeline Cost:
Day 1-2:    Transcribe 3 sample docs           (4-6 hours)
Day 2:      Run OCR + verify integration       (0.5-1 hour)
Day 3:      Analyze & decide go/no-go         (2-3 hours)
            Subtotal: 6-9 hours ✅

If successful:
Day 4-6:    Expand to 50+ documents           (24-32 hours)

If failed:
Day 4:      Fix minimum issue                  (2-4 hours)
Day 5:      Rerun smoke test                   (1 hour)
            Then proceed to expansion

Worst case: 9-13 hours + fix time
Best case: 9 hours + expanded validation
```

---

## Smoke Test Corpus (Finalized)

### Selection Criteria

✅ **Real representative samples** (not synthetic)  
✅ **Small enough to execute quickly** (6-9 hours turnaround)  
✅ **Academic content** (BUILD's actual use case)  
✅ **Mix of formats** (images + PDF)  
✅ **Rich structure** (equations, MCQs, headings for qualitative assessment)  

### Documents Chosen

| # | File | Format | Size | Representation |
|---|------|--------|------|-----------------|
| 1 | DTS SECOND SEMESTER EXAMINATION.jpeg | Image | 129 KB | Exam paper (structured) |
| 2 | IMG-20260627-WA0036.jpg | Image | 118 KB | Mobile capture (real-world) |
| 3 | MTS 102 Multiple Choice Limits.pdf | PDF | 4.45 MB | Academic MCQ content (scanned) |

**PDF Strategy**: Sample 3 pages (1, 10, 18) to test different content regions:
- Page 1: Title, introduction, early questions
- Page 10: Mid-document questions (stable content)
- Page 18: Final page (end-of-document behavior)

**Total annotation burden**: ~4-5 pages of actual text (not 18 full pages)

---

## Critical Requirement: Use Actual App OCR Service

### ✅ Correct Path (What We're Doing)

```
JPEG/PDF Document
    ↓
app/core/documents/intake.py
    Validates tenant context, generates document ID
    ↓
app/core/documents/analysis.py
    Classifies page types (NATIVE_TEXT vs SCANNED vs MIXED)
    ↓
app/core/documents/ocr.py::OCRRouter
    Routes based on classification
    ↓
app/services/ocr/ (Production OCR service)
    Selects OCRProvider implementation
    ↓
app/services/ocr/...provider.py::PaddleOCRProvider
    Wraps raw SDK, normalizes to OCRResult
    ↓
app/core/documents/openvino.py::OpenVINORuntime
    Compiles + runs model on device (CPU/GPU/NPU)
    ↓
PP-OCRv3 Detection Model
    Finds text regions, bounding boxes
    ↓
PP-OCRv3 Recognition Model
    Recognizes characters in each region
    ↓
OCRResult (contract)
    text: str, regions: list[OCRTextRegion]
    ↓
Compare to Ground Truth
    CER/WER calculated via app/core/documents/metrics.py
```

**Why This Matters**:
- Proves the production path works
- Captures real integration issues (if any)
- Results are representative of actual BUILD OCR behavior
- Baseline is repeatable using the same path

### ❌ Wrong Path (What We're NOT Doing)

```
JPEG/PDF Document
    ↓
Standalone benchmark script
    Reads file directly (no intake validation)
    ↓
Call PaddleOCR SDK directly (no abstraction)
    Bypasses OCRProvider pattern
    Bypasses OpenVINORuntime checks
    ↓
Metrics
    Result: Numbers don't represent production behavior
    Risk: "It works in benchmark but fails in production"
```

---

## Ground Truth Annotation Standards

### What to Transcribe

**Transcribe**: Exactly what you see in the source document
- All readable text
- All visible numbers, symbols, formatting
- Question numbers, option letters (A/B/C/D)
- Section headings and subheadings

**Do NOT transcribe**:
- Handwritten notes or marks (mark as `[handwritten]`)
- Diagrams or charts (mark as `[diagram]`)
- Extremely faint/illegible text (mark as `[illegible]`)

### Format Rules

```json
{
  "document": {
    "filename": "DTS SECOND SEMESTER EXAMINATION.jpeg",
    "document_type": "examination_paper",
    "content_type": "raster_image",
    "page_count": 1
  },
  "ground_truth": {
    "full_text": "DEPARTMENT OF EDUCATION\nDTS SECOND SEMESTER EXAMINATION\n\nCourse: Mathematics\nTime: 2 hours\n\n1. What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6\n\n2. Solve for x: x² + 3x - 4 = 0\nA. x = 1 or x = -4\nB. x = -1 or x = 4\nC. [illegible]\nD. None\n\n[Continue for all visible text]"
  }
}
```

### Validation

After creating each JSON file:
```bash
python -m json.tool data/ocr_evaluation/ground_truth/FILE.json > /dev/null
echo $?  # Should output 0 (valid JSON)
```

---

## Expected Outcomes

### Quantitative Metrics (Real Measurements)

For each page, we will capture:

| Metric | Description | Example |
|--------|-------------|---------|
| **CER** | Character Error Rate | 3.2% |
| **WER** | Word Error Rate | 6.8% |
| **Latency** | Time to OCR one page | 850 ms |
| **Throughput** | Pages processed per second | 1.18 pages/sec |
| **Memory Peak** | Maximum memory used | 481 MB |
| **Success** | Did OCR complete without error? | true/false |

**Aggregate** (across 3 documents):
```
Avg CER:      3.8%
Avg WER:      7.6%
Mean Latency: 950 ms/page
Throughput:   1.05 pages/sec
Peak Memory:  512 MB
```

### Qualitative Assessment (Critical for Educational Content)

Inspect OCR outputs for:

| Aspect | What to Check | Examples of Issues |
|--------|---------------|-------------------|
| **Equations** | Are math symbols preserved? | x² → x2 (loss of meaning) |
| **MCQ Format** | Are A/B/C/D associations clear? | "A. 15 B. 25" correctly parsed? |
| **Headings** | Are titles/sections recognized? | "Section 1: Algebra" → ignored? |
| **Numbering** | Are lists intact? | "1. 2. 3." vs "1 2 3"? |
| **Superscripts** | Are x^n preserved? | H₂O vs H2O? |
| **Symbols** | Are special chars handled? | θ, ∆, ∑, ∫ → corrupted? |
| **Layout** | Is reading order correct? | Multi-column: left-to-right? |

**Example Finding**:
> "MCQ options are well-preserved. Equations show moderate degradation (fractions as division slash). Overall structure maintained. Layout order correct. Superscripts on Page 10 need review."

---

## Success Criteria

### Smoke Test Passes ✅ When:
- All 3 documents process without unrecoverable errors
- OCR produces text output (not empty)
- CER/WER calculated from real ground truth comparison
- Metrics are reproducible across runs
- Qualitative assessment completed
- No fabricated numbers in report
- Integration path confirmed working

### Then We Can ✅:
1. Proceed to corpus expansion (50+ documents)
2. Acquire and test newer PaddleOCR versions
3. Acquire and test Surya model
4. Run identical benchmark on all candidates
5. Compare and select production model

### If Smoke Test Fails ❌:
1. Do NOT expand scope to new models yet
2. Diagnose the minimum issue (file path? model path? memory? OpenVINO? integration?)
3. Fix only that issue
4. Rerun smoke test
5. Validate fix
6. Then proceed to expansion

---

## Real vs Estimated Metrics

### ✅ REAL (Good)

```json
{
  "cer": 3.2,           // Calculated from actual ground truth comparison
  "latency_ms": 1205,   // Measured from actual execution
  "pages_per_second": 0.83,  // 1000 / 1205
  "peak_memory_mb": 512  // Observed during execution
}
```

### ❌ ESTIMATED (Bad)

```json
{
  "cer": 3.0,           // "Typical value for PP-OCRv3"
  "latency_ms": 1000,   // Rounded for convenience
  "pages_per_second": 1.0,  // Theoretical throughput
  "peak_memory_mb": 500  // "Probably around this"
}
```

**User Principle**: "No fabricated metrics. Every number must come from actual measured OCR execution."

---

## Timeline & Roles

### Ground Truth Annotation (4-6 hours)
**Owner**: BUILD AI/ML Team  
**Tasks**:
- [ ] Examine 2 images visually, transcribe completely
- [ ] Examine PDF pages 1, 10, 18, transcribe completely
- [ ] Validate JSON syntax
- [ ] Verify transcriptions match source exactly

### OCR Execution (0.5-1 hour)
**Owner**: Build/DevOps + AI/ML Team  
**Tasks**:
- [ ] Verify OpenVINO + PaddleOCR installed (`pip install -r requirements.txt`)
- [ ] Verify model files in `models/ocr/openvino_models/`
- [ ] Run `python scripts/smoke_test_benchmark.py`
- [ ] Monitor for errors
- [ ] Capture actual OCR text output

### Analysis & Reporting (1-2 hours)
**Owner**: AI/ML Team  
**Tasks**:
- [ ] Load metrics from `results/smoke_test_baseline.json`
- [ ] Inspect OCR outputs qualitatively
- [ ] Create `results/SMOKE_TEST_REPORT.md`
- [ ] Document findings and recommendations

### Decision (30 min)
**Owner**: Build Leadership  
**Criteria**:
- [ ] All 3 documents processed? ✅ → Proceed to expansion
- [ ] Integration issues? ❌ → Fix minimum issue, retry

---

## Reference Documents

| File | Purpose |
|------|---------|
| `data/ocr_evaluation/SMOKE_TEST_BASELINE.md` | Detailed execution guide (6 phases) |
| `scripts/smoke_test_benchmark.py` | Benchmark harness template (async-ready) |
| `app/core/documents/metrics.py` | CER/WER calculation (15+ tests passing) |
| `app/core/documents/openvino.py` | OpenVINO runtime (20 tests passing) |
| `app/services/ocr/` | Production OCR service abstraction |

---

## Next Action

**For BUILD Team**:

1. **Phase 0.2** → Read `data/ocr_evaluation/SMOKE_TEST_BASELINE.md`
2. **Phase 0.2** → Annotate 2 images + 3 PDF pages (4-6 hours)
3. **Phase 0.3** → Run `python scripts/smoke_test_benchmark.py`
4. **Phase 0.4** → Capture real metrics
5. **Phase 0.5** → Inspect outputs qualitatively
6. **Phase 0.6** → Produce baseline report
7. **Decision** → Approve corpus expansion or diagnose issues

---

## Success Message

Once completed, the report should state:

> **PP-OCRv3 / OpenVINO Smoke Test Baseline**
>
> ✅ Status: Success  
> ✅ Documents tested: 3 (2 images, 1 PDF)  
> ✅ All documents processed successfully  
>
> **Quantitative Results**:
> - Avg CER: X.X%
> - Avg WER: X.X%
> - Mean latency: XXX ms/page
> - Throughput: X.XX pages/sec
> - Peak memory: XXX MB
>
> **Qualitative Assessment**:
> - MCQ format preserved ✅
> - Equations: [observations]
> - Symbols: [observations]
> - Layout: [observations]
>
> **Recommendation**: ✅ Proceed to corpus expansion

---

**Created**: 2026-09-01  
**Version**: 1.0 (Smoke Test Baseline)  
**Status**: Ready for annotation phase
