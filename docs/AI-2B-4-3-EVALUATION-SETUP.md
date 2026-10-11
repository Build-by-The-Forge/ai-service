# AI-2B.4.3 Setup Complete: Evaluation-First OCR Architecture

> **2026-10-11:** the model files this document refers to (`*_infer/`, `*.tar`, `*.onnx`, `models/ocr/openvino_models/`) were removed from the repository. Models now come only through `models/ocr/manifest.json` and `scripts/fetch_ocr_models.py` (checksum-verified). The OpenVINO IR conversions are no longer tracked; regenerate them from the fetched models for benchmarking.

**Date:** 2026-09-01  
**Status:** ✅ Infrastructure ready for OCR model benchmarking  
**Next Phase:** Evidence-based model selection before implementation

---

## What Was Established

We have moved from a premature implementation-first approach to a **production-engineering-first approach**:

### Before (Risk)
```
PP-OCRv3 artifacts exist
    ↓
Start DBNet preprocessing implementation
    ↓
Months of work
    ↓
Discover newer model is 10% more accurate
    ↓
Rework (expensive)
```

### Now (Proven Strategy)
```
OCR candidates identified
    ↓
Create evaluation corpus (images + PDFs + ground truth)
    ↓
Benchmark all candidates on same data
    ↓
Evidence-based selection
    ↓
Implement winning model only
    ↓
No regret, high confidence
```

---

## Infrastructure Created

### 1. Evaluation Layer

**File:** [app/core/documents/evaluation.py](app/core/documents/evaluation.py)

This module keeps model selection **separate from runtime**:

```python
class OCRModelCandidate:
    """Model metadata: name, family, deployment fit, licensing"""
    name: str
    family: str
    openvino_supported: bool
    document_oriented: bool
    multilingual_support: str
    licensing: str
    status: str  # baseline, candidate, deferred

class OCRModelEvaluator:
    """Ranks candidates by weighted score"""
    def evaluate_candidates(...) -> list[OCRCandidateEvaluation]
    def select_primary_candidate(...) -> OCRModelCandidate
```

**Tests:** [tests/unit/test_document_ocr_evaluation.py](tests/unit/test_document_ocr_evaluation.py) (3 passing)

**Key point:** The evaluator is intentionally **model-agnostic**. It lets evidence drive decisions, not assumptions.

### 2. Evaluation Dataset Structure

**Location:** [data/ocr_evaluation/](data/ocr_evaluation/)

```
data/ocr_evaluation/
├── README.md                    ← Overview and usage
├── BENCHMARKING_CHECKLIST.md    ← Step-by-step execution plan
├── DECISION_FRAMEWORK.md        ← Scoring rubric (post-benchmark)
│
├── images/                      ← Image test corpus (to populate)
│   ├── id_card.png
│   ├── passport.jpg
│   ├── certificate.jpg
│   ├── academic_transcript.jpg
│   ├── lecture_notes.png
│   ├── form_with_tables.png
│   ├── mathematical_text.jpg
│   ├── low_quality.png
│   ├── rotated_page.png
│   └── skewed_page.png
│
├── pdfs/                        ← PDF test corpus (to populate)
│   ├── academic_transcript.pdf
│   ├── scanned_document.pdf
│   ├── mixed_document.pdf
│   ├── form_with_tables.pdf
│   ├── multi_column_notes.pdf
│   └── rotated_pages.pdf
│
└── ground_truth/                ← JSON annotations (to populate)
    ├── id_card.json
    ├── passport.json
    ├── certificate.json
    └── ...
```

### 3. Benchmarking Plan

**File:** [data/ocr_evaluation/BENCHMARKING_CHECKLIST.md](data/ocr_evaluation/BENCHMARKING_CHECKLIST.md)

A 7-part execution plan:

1. **Baseline Validation** — Confirm PP-OCRv3 baseline is ready
2. **Candidate Acquisition** — Get Newer PaddleOCR and Surya models
3. **Corpus Building** — Collect representative BUILD documents
4. **Infrastructure** — Write benchmark and comparison scripts
5. **Benchmark Execution** — Run all three candidates
6. **Decision Framework** — Score and select winner
7. **Implementation** — Build winning architecture (post-selection)

### 4. Decision Framework

**File:** [data/ocr_evaluation/DECISION_FRAMEWORK.md](data/ocr_evaluation/DECISION_FRAMEWORK.md)

A structured rubric for comparing candidates across:

- **Text Accuracy (40%)** — WER, CER, confidence
- **Performance (15%)** — Latency, memory, model size
- **Deployment Fit (20%)** — OpenVINO support, stability
- **Language Coverage (10%)** — Multilingual capability
- **Licensing (10%)** — Production legality
- **Community (5%)** — Long-term support

Each category is scored post-benchmark. Weights reflect BUILD's priorities: **accuracy > deployment > performance > language > licensing > community**.

---

## Architecture Decision

We are **NOT locking BUILD to PP-OCRv3 prematurely**.

Instead:

```
        OCRProvider (BUILD contract)
             │
      ┌──────┼──────┐
      │      │      │
  Baseline  Candidate  Candidate
   (PP-v3)  (New PL)   (Surya)
      │      │      │
      └──────┼──────┘
             │
     Benchmark results
             │
             ▼
   Select primary + fallback
             │
             ▼
  Implement only winner
             │
             ▼
 OCRProvider stays replaceable
```

The provider boundary remains unchanged. Whichever model wins, it plugs into the existing contract.

---

## Verification Status

### Test Results

✅ **Document Module Regression:** 27 passing tests
- AI-2B.1 contracts: 6 tests
- AI-2B.2 intake: 5 tests
- AI-2B.3 analysis: 6 tests
- AI-2B.4.1 OCR contracts: 6 tests
- AI-2B.4.3 evaluation: 3 tests (NEW)

✅ **OpenVINO Runtime:** 20 passing tests
- Initialization, device discovery, path validation
- Model compilation, caching, lifecycle
- Error handling (typed exceptions)

✅ **Code Quality:** Ruff linting passes
- New evaluation module: clean
- No import issues, type hints respected

---

## Next Steps

### Immediate (Pre-Benchmarking)

1. **Baseline Validation** (1-2 days)
   - [ ] Verify PP-OCRv3 models are ready
   - [ ] Document expected performance baselines
   - [ ] Tag as `ppocr_v3_baseline`

2. **Candidate Acquisition** (3-5 days)
   - [ ] Research newer PaddleOCR releases (v3+)
   - [ ] Get Surya models and licensing details
   - [ ] Validate OpenVINO export paths
   - [ ] Store models in `models/ocr/openvino_models/`

### Short-term (Benchmarking)

3. **Build Evaluation Corpus** (5-10 days)
   - [ ] Collect 50-100 representative documents
   - [ ] Create ground truth annotations (JSON)
   - [ ] Ensure anonymization (no real PII)

4. **Benchmarking Infrastructure** (3-5 days)
   - [ ] `scripts/benchmark_ocr.py` — run each model
   - [ ] `scripts/compare_ocr_results.py` — generate report

5. **Run Benchmarks** (2-3 days)
   - [ ] Execute against all candidates
   - [ ] Capture metrics (accuracy, speed, memory, etc.)
   - [ ] Generate comparison report

6. **Make Decision** (1 day)
   - [ ] Fill in `DECISION_FRAMEWORK.md`
   - [ ] Score candidates
   - [ ] Document rationale
   - [ ] Select primary + fallback

### Medium-term (Post-Decision)

7. **Implementation** (Timeline depends on winning model)
   - Only after benchmark confirms winner
   - Implement OCR adapter behind existing provider boundary
   - Add preprocessing, postprocessing, orchestration

---

## Key Architectural Principles Preserved

✅ **Replaceable OCRProvider** — whichever model wins, it plugs in the same way  
✅ **Document Intake boundary** — unchanged  
✅ **Document Analysis** — still model-agnostic  
✅ **OpenVINO runtime** — proven and generic  
✅ **No model lock-in** — evidence drives the choice  

---

## Why This Matters for BUILD

**Educational documents demand accuracy.** An ID card, transcript, or certificate misrecognized is a problem.

By benchmarking **before** implementing, we:

1. **Know exactly** which model is best for BUILD's documents
2. **Avoid rework** if a better option emerges
3. **Keep decisions evidence-based**, not assumption-based
4. **Preserve architecture** even when swapping models
5. **Maintain confidence** that we chose the right tool

---

## Related Files

- Milestone status: [/memories/repo/ai-2b-milestone-status.md](/memories/repo/ai-2b-milestone-status.md)
- Evaluation module: [app/core/documents/evaluation.py](app/core/documents/evaluation.py)
- Evaluation tests: [tests/unit/test_document_ocr_evaluation.py](tests/unit/test_document_ocr_evaluation.py)
- Benchmarking guide: [data/ocr_evaluation/BENCHMARKING_CHECKLIST.md](data/ocr_evaluation/BENCHMARKING_CHECKLIST.md)
- Decision rubric: [data/ocr_evaluation/DECISION_FRAMEWORK.md](data/ocr_evaluation/DECISION_FRAMEWORK.md)

---

## Success Criteria

✅ Infrastructure is ready  
✅ Evaluation layer is tested  
✅ Benchmarking plan is documented  
✅ Decision framework is defined  
✅ Architecture remains replaceable  
✅ No models locked in yet  

**Next:** Follow the checklist and execute benchmarking.

---

**Prepared by:** AI Service Engineering Team  
**Phase:** AI-2B.4.3 Evaluation (Infrastructure)  
**Status:** READY FOR BENCHMARKING
