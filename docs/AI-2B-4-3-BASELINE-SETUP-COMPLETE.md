# OCR Baseline Benchmarking: Infrastructure Setup Complete

**Status**: ✅ Phase 1 Complete — Ready for Ground Truth Population  
**Date**: 2026-09-01  
**Next Action**: Populate ground truth JSON files for all 7 documents  

---

## What We've Built

### 1. ✅ Ground Truth Templates (7 documents)

**Location**: `data/ocr_evaluation/ground_truth/`

Templates created for each document with metadata and structure:

| Document | File | Format | Pages | Size |
|----------|------|--------|-------|------|
| DTS Exam | `DTS_SECOND_SEMESTER_EXAMINATION.json` | Image | 1 | 129 KB |
| Mobile Doc | `IMG-20260627-WA0036.json` | Image | 1 | 118 KB |
| MTS Limits MCQ | `MTS_102_Multiple_Choice_Limits.json` | PDF | 18 | 4.5 MB |
| MTS Differentiation | `MTS_102_Differentiation_Exercise.json` | PDF | 2 | 43 KB |
| MTS Cheat Sheet | `MTS_102_Cheat_Sheet.json` | PDF | 4 | 135 KB |
| MTS Solutions | `MTS102_Solutions.json` | PDF | ? | 247 KB |
| PHY 102 MCQ | `PHY_102_Practice_MCQ.json` | PDF | 21 | 647 KB |

Each template includes:
- Document metadata (type, source, quality notes)
- `ground_truth.full_text` field (to be filled with expected OCR output)
- Optional text regions for fine-grained analysis
- Expected challenges and difficulty notes

### 2. ✅ Benchmark Harness Script

**Location**: `scripts/benchmark_ocr_baseline.py`

A production-ready benchmark script that:
- Discovers all documents in corpus (images + PDFs)
- Processes each document through OCR pipeline
- Collects per-document metrics:
  - CER (Character Error Rate)
  - WER (Word Error Rate)
  - Latency (ms/page, total)
  - Memory (peak RAM)
  - Throughput (pages/second)
  - Detection quality, layout preservation
- Aggregates results across corpus
- Exports results to JSON with full breakdown

**Features**:
- Async/await for efficient processing
- Structured data classes (DocumentMetrics, BenchmarkResult)
- OpenVINO device discovery and selection
- Error handling and failure categorization
- Logging with per-document status
- Extensible for multi-candidate comparison

**Usage**:
```bash
python scripts/benchmark_ocr_baseline.py \
    --corpus-dir data/ocr_evaluation \
    --output results/baseline_ppocr_v3.json \
    --candidate ppocr_v3 \
    --device CPU
```

### 3. ✅ Metrics Calculation Module

**Location**: `app/core/documents/metrics.py`

Implements standard OCR accuracy metrics:

- **CER (Character Error Rate)**: Levenshtein distance at character level
  - Formula: (insertions + deletions + substitutions) / reference_length × 100%
  - Tests passing: ✓ Perfect match (0%), ✓ Edits (20-200%), ✓ Edge cases
  
- **WER (Word Error Rate)**: Levenshtein distance at word level
  - Formula: (word insertions + deletions + substitutions) / reference_words × 100%
  - Tests passing: ✓ Perfect match (0%), ✓ Multi-word sequences

- **GroundTruthLoader**: Loads and parses JSON ground truth files
  - Extracts full text for CER/WER calculation
  - Extracts text regions for fine-grained analysis
  - Handles multi-page PDFs and single images

- **ComparisonResult**: Dataclass for OCR vs Ground Truth comparison
  - CER and WER with edit operation breakdown
  - Document metadata and ground truth availability flag

**Validation Status**:
```
✓ 5 test categories (15+ assertions)
✓ Perfect match scenarios
✓ Character/word error variants
✓ Complex multi-line text
✓ Edge cases (empty strings, 100% errors)
✓ OCR-realistic scenarios
```

### 4. ✅ Execution Guide

**Location**: `data/ocr_evaluation/BASELINE_EXECUTION_GUIDE.md`

Comprehensive 4-phase guide:

- **Phase 1: Ground Truth Population**
  - What is ground truth and why it matters
  - Step-by-step instructions for each document
  - Expected time estimates (12-18 hours for 7 documents)
  - Automation suggestions (cloud OCR + human review)
  - Multi-page PDF handling strategies

- **Phase 2: Benchmark Execution**
  - Prerequisites (OpenVINO, PaddleOCR, models)
  - Running the benchmark script
  - Expected output format and structure
  - Interpreting CER/WER results
  - Performance baseline reference

- **Phase 3: Results Analysis**
  - Creating summary report template
  - Failure analysis and troubleshooting
  - Common error types and fixes

- **Phase 4: Expand Corpus** (optional)
  - Adding more documents after baseline validation
  - Ground truth and benchmarking for expanded set

### 5. ✅ Evaluation Infrastructure (Pre-existing)

From previous sessions:

- **Evaluation Module**: `app/core/documents/evaluation.py`
  - OCR model candidate ranking
  - Multi-candidate comparison framework
  - Weighted scoring formula (accuracy 80%, latency 10%, memory 5%, languages 8%, bonuses)
  
- **Decision Framework**: `data/ocr_evaluation/DECISION_FRAMEWORK.md`
  - Scoring rubric for post-benchmark comparison
  - Candidate evaluation metrics
  - Trade-off documentation
  
- **Benchmarking Checklist**: `data/ocr_evaluation/BENCHMARKING_CHECKLIST.md`
  - 7-step execution plan
  - Timeline estimates
  - Task tracking

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      BUILD AI Service                        │
│                    OCR Baseline Workflow                      │
└─────────────────────────────────────────────────────────────┘
                                
┌──────────────┐          ┌──────────────┐        ┌──────────────┐
│   Corpus     │          │   Ground     │        │  Benchmark   │
│  Documents   │─────────→│   Truth      │───────→│   Harness    │
│              │          │   Templates  │        │              │
└──────────────┘          └──────────────┘        └──────────────┘
    7 files:                7 JSON files:         benchmark_ocr_
    - 2 images               - Fill in expected    baseline.py
    - 5 PDFs                   OCR output
                                                    ↓
                            ┌──────────────────────────────┐
                            │  OCR Pipeline Execution      │
                            │  (OpenVINO + PaddleOCR)     │
                            │                              │
                            │  For each document:          │
                            │  - Load image/PDF            │
                            │  - Run OCR                   │
                            │  - Measure latency/memory    │
                            │  - Compare to ground truth   │
                            │  - Calculate CER/WER         │
                            └──────────────────────────────┘
                                    ↓
                            ┌──────────────────────────────┐
                            │   Metrics Calculation        │
                            │   (metrics.py)               │
                            │                              │
                            │  - CER (character errors)    │
                            │  - WER (word errors)         │
                            │  - Latency breakdown         │
                            │  - Memory peak usage         │
                            │  - Throughput               │
                            └──────────────────────────────┘
                                    ↓
                            ┌──────────────────────────────┐
                            │   Results Aggregation        │
                            │   (benchmark script)         │
                            │                              │
                            │  - Per-document metrics      │
                            │  - Aggregate statistics      │
                            │  - Failure summary           │
                            │  - JSON export               │
                            └──────────────────────────────┘
                                    ↓
                        ┌──────────────────────┐
                        │  Results JSON        │
                        │  baseline_ppocr_v3.  │
                        │  json                │
                        └──────────────────────┘
```

---

## File Structure

```
ai-service/
├── data/ocr_evaluation/
│   ├── images/
│   │   ├── DTS SECOND SEMESTER 20252026 EXAMINATION.jpeg
│   │   └── IMG-20260627-WA0036.jpg
│   ├── pdfs/
│   │   ├── MTS 102  Multiple Choice Questions on Limits...pdf
│   │   ├── MTS 102 - Differentiation Exercise...pdf
│   │   ├── MTS 102 Cheat sheet.pdf
│   │   ├── MTS102_Solutions.pdf
│   │   └── PHY 102 PRACTICE MULTIPLE CHOICE QUESTIONS...pdf
│   ├── ground_truth/  ← TO BE FILLED
│   │   ├── DTS_SECOND_SEMESTER_EXAMINATION.json (template)
│   │   ├── IMG-20260627-WA0036.json (template)
│   │   ├── MTS_102_Multiple_Choice_Limits.json (template)
│   │   ├── MTS_102_Differentiation_Exercise.json (template)
│   │   ├── MTS_102_Cheat_Sheet.json (template)
│   │   ├── MTS102_Solutions.json (template)
│   │   └── PHY_102_Practice_MCQ.json (template)
│   ├── README.md (existing)
│   ├── BENCHMARKING_CHECKLIST.md (existing)
│   ├── DECISION_FRAMEWORK.md (existing)
│   └── BASELINE_EXECUTION_GUIDE.md (NEW)
├── scripts/
│   ├── benchmark_ocr_baseline.py (NEW)
│   └── [existing scripts]
├── app/
│   └── core/
│       └── documents/
│           ├── evaluation.py (existing)
│           ├── metrics.py (NEW)
│           ├── openvino.py (existing)
│           ├── ocr.py (existing)
│           ├── analysis.py (existing)
│           ├── intake.py (existing)
│           ├── contracts.py (existing)
│           └── __init__.py
├── models/ocr/openvino_models/
│   ├── det/
│   │   ├── en_PP-OCRv3_det_infer.xml
│   │   └── en_PP-OCRv3_det_infer.bin
│   └── rec/
│       ├── ch_PP-OCRv3_rec_infer.xml
│       └── ch_PP-OCRv3_rec_infer.bin
└── results/
    ├── baseline_ppocr_v3_cpu.json (will be created)
    ├── baseline_execution.log (will be created)
    └── BASELINE_ANALYSIS.md (to be created)
```

---

## Next Steps (Immediate)

### Priority 1: Ground Truth Population (BLOCKING)

**Estimated Time**: 12-18 hours

For each document in `data/ocr_evaluation/ground_truth/`:

1. **Open the source document** (image or PDF)
2. **Manually transcribe all readable text** to `ground_truth.full_text` in JSON
3. **Save and verify** JSON is valid

**Documents in priority order**:

1. `DTS_SECOND_SEMESTER_EXAMINATION.json` — 1-2 hours (1-page image)
2. `IMG-20260627-WA0036.json` — 0.5-1 hour (1-page image)
3. `MTS_102_Multiple_Choice_Limits.json` — 3-5 hours (18-page PDF)
4. `MTS_102_Differentiation_Exercise.json` — 1 hour (2-page PDF)
5. `PHY_102_Practice_MCQ.json` — 3-5 hours (21-page PDF)

**Optional (lower priority)**:
- `MTS_102_Cheat_Sheet.json` — 1-2 hours
- `MTS102_Solutions.json` — 2-3 hours

### Priority 2: Benchmark Execution (AFTER Ground Truth)

**Requirements**:
- OpenVINO 2023.0+ installed
- PaddleOCR 2.7.0+ installed
- All dependencies in requirements.txt
- Ground truth 100% populated

**Command**:
```bash
python scripts/benchmark_ocr_baseline.py \
    --corpus-dir data/ocr_evaluation \
    --output results/baseline_ppocr_v3_cpu.json
```

**Expected time**: 5-30 minutes depending on document sizes and device

### Priority 3: Results Analysis & Report (AFTER Execution)

- ✅ Verify success rate (>95%)
- ✅ Check CER/WER (<5%/<10% targets)
- ✅ Create summary report
- ✅ Document findings and recommendations

---

## Success Criteria

### Baseline Validation Success

✅ All 7 documents process without error (100% success rate)  
✅ Average CER < 5%  
✅ Average WER < 10%  
✅ Latency < 2000 ms/page on CPU  
✅ Memory usage < 1 GB on CPU  
✅ Reproducible results on multiple runs  
✅ Ground truth fully populated for all documents  

### Metrics Test Coverage

✅ 15+ unit test assertions passing  
✅ CER calculation correct (Levenshtein distance)  
✅ WER calculation correct (word-level Levenshtein)  
✅ Edge cases handled (empty strings, 100% error, etc.)  
✅ JSON ground truth parsing works  

---

## Decision Points & Trade-offs

### Why Baseline First?

**Before**: Build infrastructure → implement candidate model → benchmark → discover it's not the best → rework

**Now**: Baseline PP-OCRv3 → validate path works → expand corpus → test candidates → evidence-based selection

**Benefit**: Avoid premature implementation; save rework time

### Why These 7 Documents?

- **Representative of BUILD's domain**: Educational materials (exams, exercises, solutions)
- **Diverse content types**: 
  - Plain scanned images (2)
  - Mathematical content with equations (4)
  - Dense formatting with multiple columns (1)
  - Physics with scientific notation (1)
- **Small enough for quick baseline**: ~5.8 MB total, 48 pages
- **Large enough to be meaningful**: Not just toy examples

### Why Not Multi-Candidate Immediately?

**Risk**: Benchmark newer models before validating execution path works
**Mitigation**: Baseline first proves the pipeline; then safe to test alternatives

---

## Troubleshooting Reference

| Issue | Solution |
|-------|----------|
| OpenVINO not found | Install: `pip install openvino>=2023.0` |
| PaddleOCR not found | Install: `pip install paddleocr>=2.7.0` |
| Model files missing | Check `models/ocr/openvino_models/det/` and `rec/` |
| Ground truth file invalid JSON | Validate JSON: `python -m json.tool file.json` |
| High CER/WER (>15%) | Check ground truth accuracy; document may be low-quality |
| Timeout on large PDF | Increase timeout; consider sampling pages |
| Out of memory | Use GPU device instead of CPU |

---

## Metrics Reference

### CER/WER Interpretation

| Range | Assessment | Action |
|-------|------------|--------|
| 0-2% | Excellent | ✓ Approve for production |
| 2-5% | Good | ✓ Acceptable, minor review |
| 5-10% | Fair | ⚠️ Acceptable with review |
| 10-15% | Poor | ⚠️ Needs investigation |
| >15% | Very Poor | ✗ Reject; requires refinement |

### Latency Baseline (CPU)

| Range | Assessment |
|-------|------------|
| <500ms | Excellent (real-time capable) |
| 500-1000ms | Good (acceptable for batch) |
| 1000-2000ms | Fair (slower batch processing) |
| >2000ms | Poor (needs optimization) |

---

## Dependencies

### Required (for benchmark execution)

```
openvino>=2023.0          # OpenVINO inference runtime
paddleocr>=2.7.0          # PaddleOCR models and API
Pillow>=9.0.0             # Image processing
pydantic>=2.5.0           # Data validation
```

### Already Available

```
pytest                    # Testing framework
asyncio                   # Async execution
dataclasses               # Type-safe data structures
pathlib                   # File path handling
logging                   # Logging infrastructure
json                      # JSON serialization
```

---

## Appendix: Ground Truth Entry Example

**Document**: MTS 102 - Differentiation Exercise (2-page PDF)

```json
{
  "document": {
    "filename": "MTS 102 - Differentiation Exercise  BY PHOTOCHEM.pdf",
    "document_type": "exercise_sheet",
    "content_type": "pdf_scanned",
    "page_count": 2,
    "file_size_mb": 0.043
  },
  "ground_truth": {
    "full_text": "MTS 102 - Differentiation Exercise\nBy: Photochem\n\nINSTRUCTIONS:\nSolve the following differentiation problems. Show all working.\n\nEXERCISE 1:\nFind the derivative of the following functions:\n\n1. f(x) = 3x² + 2x - 5\n   Solution: f'(x) = 6x + 2\n\n2. f(x) = (x + 1)(x - 2)\n   Solution: f'(x) = 2x - 1\n\n3. f(x) = 1/x\n   Solution: f'(x) = -1/x²\n\nEXERCISE 2:\n[Continue with page 2 content...]"
  },
  "metadata": {
    "created_date": "2026-09-01",
    "ground_truth_type": "full_manual_transcription",
    "pages_transcribed": [1, 2]
  }
}
```

---

## Summary

We have built a **complete baseline benchmarking infrastructure** for OCR model evaluation. The pipeline is:

1. ✅ **Structured** (templates, harness, metrics module all in place)
2. ✅ **Validated** (metrics calculation tested, edge cases handled)
3. ✅ **Documented** (4-phase execution guide, troubleshooting reference)
4. ✅ **Ready** (just needs ground truth population to execute)

The next action is **manual ground truth population** for 7 documents, which will unblock full baseline execution and produce real measurable metrics (CER, WER, latency, memory).

After baseline succeeds, the path is clear for:
- Expanded corpus testing
- Multi-candidate benchmarking
- Evidence-based model selection
- Production implementation

---

**Owner**: BUILD AI/ML Team  
**Created**: 2026-09-01  
**Status**: Ready for execution  
**Next Review**: After ground truth population complete
