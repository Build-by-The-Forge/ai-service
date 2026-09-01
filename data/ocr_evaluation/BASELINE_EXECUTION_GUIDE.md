# OCR Baseline Benchmark: Execution Guide

**Status**: Baseline Phase Initialized  
**Objective**: Execute PP-OCRv3/OpenVINO against real BUILD documents and produce measurable CER/WER/latency metrics  
**Timeline**: 3-5 business days for full baseline execution  

---

## Overview

This guide walks through the **baseline benchmarking phase** for OCR model selection in BUILD. We have:

- ✅ **7 real documents** in evaluation corpus (2 images, 5 PDFs)
- ✅ **Ground truth templates** created in `data/ocr_evaluation/ground_truth/`
- ✅ **Benchmark harness** ready in `scripts/benchmark_ocr_baseline.py`
- ✅ **Metrics calculation** module ready in `app/core/documents/metrics.py`
- ⏳ **Ground truth annotation** (NEXT STEP - must be done manually)
- ⏳ **Benchmark execution** (NEXT STEP - must run on machine with OpenVINO)
- ⏳ **Results analysis** (NEXT STEP)

---

## Phase 1: Ground Truth Population

### What is Ground Truth?

Ground truth is the **expected correct OCR output** for each document. It's used to:
- Calculate Character Error Rate (CER) and Word Error Rate (WER)
- Compare OCR output against a known-correct baseline
- Measure accuracy of the OCR engine

### Ground Truth Format

Each document has a corresponding JSON file in `data/ocr_evaluation/ground_truth/`:

```json
{
  "document": {
    "filename": "DTS_SECOND_SEMESTER_EXAMINATION.jpeg",
    "document_type": "examination_paper",
    "content_type": "raster_image",
    "page_count": 1
  },
  "ground_truth": {
    "full_text": "DTS SECOND SEMESTER 2025/2026 EXAMINATION\n\n[Full transcription of exam content]\n...",
    "text_regions": [
      {
        "id": "header",
        "text": "DTS SECOND SEMESTER 2025/2026 EXAMINATION",
        "type": "heading",
        "confidence_expected": 0.95
      }
    ]
  },
  "metadata": {
    "created_date": "2026-09-01",
    "ground_truth_type": "manual_transcription"
  }
}
```

### How to Populate Ground Truth

#### For Each Document:

1. **Open the document** (image or PDF)
   - Images: Open in image viewer
   - PDFs: Open in PDF reader, or render pages to view

2. **Transcribe the full text** manually
   - Type out all readable text exactly as it appears
   - Preserve structure where meaningful (headings, questions, options, etc.)
   - If text is illegible, note `[illegible]` or `[unclear]`
   - If content is a diagram/chart (non-text), note `[diagram]`, `[chart]`, etc.

3. **Fill in the `ground_truth.full_text` field** in the JSON
   - This is the most important field for CER/WER calculation
   - Must be accurate; any transcription errors will affect metrics

4. **Optional: Add text regions** (for fine-grained analysis)
   - `text_regions` array contains individual text blocks with location/confidence estimates
   - Useful for understanding which regions had better/worse accuracy
   - Not required for basic benchmark, but recommended

5. **Save the JSON file** with the same stem as the document

#### Example Ground Truth (Image):

**File**: `data/ocr_evaluation/ground_truth/DTS_SECOND_SEMESTER_EXAMINATION.json`

```json
{
  "document": {
    "filename": "DTS SECOND SEMESTER 20252026 EXAMINATION.jpeg",
    "document_type": "examination_paper",
    "content_type": "raster_image",
    "page_count": 1,
    "resolution": "905x1280",
    "quality_notes": "Exam paper scan"
  },
  "ground_truth": {
    "full_text": "DEPARTMENT OF EDUCATION\nDTS SECOND SEMESTER 2025/2026 EXAMINATION\n\nCourse: Introduction to Mathematics\nTime: 2 hours\n\nINSTRUCTIONS:\n1. Answer all questions\n2. Show all working\n\nSECTION A: Multiple Choice\n\n1. What is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6\n\n2. Solve for x: x + 5 = 10\nA. 3\nB. 4\nC. 5\nD. 6\n\n[Continue with all visible questions...]",
    "text_regions": [
      {
        "id": "heading",
        "text": "DTS SECOND SEMESTER 2025/2026 EXAMINATION",
        "type": "heading",
        "confidence_expected": 0.95
      },
      {
        "id": "instructions",
        "text": "INSTRUCTIONS: 1. Answer all questions 2. Show all working",
        "type": "body",
        "confidence_expected": 0.90
      }
    ]
  },
  "metadata": {
    "created_date": "2026-09-01",
    "ground_truth_type": "manual_transcription",
    "difficulty_notes": "Exam paper with clear text",
    "expected_challenges": ["multiple columns possible", "varying text sizes"]
  }
}
```

#### For PDFs: Multi-Page Handling

For PDFs with multiple pages, you have two options:

**Option A: Full Transcription** (recommended for baseline)
- Transcribe all pages
- Set `page_count` to actual page count
- Combine all text into single `full_text` field with page breaks (`\n---PAGE N---\n`)

**Option B: Sampling** (if document is very long)
- Transcribe representative sample (e.g., pages 1, middle, last)
- Add note to metadata: "Sample transcription: pages 1, 10, 18 of 18"
- Still calculate metrics on all pages (unsampled pages get no GT comparison)

**Example for PDF**:

```json
{
  "document": {
    "filename": "MTS 102  Multiple Choice Questions on Limits and Continuity BY ROTIMI.pdf",
    "document_type": "course_material",
    "content_type": "pdf_scanned",
    "page_count": 18,
    "file_size_mb": 4.5
  },
  "ground_truth": {
    "full_text": "MTS 102: MULTIPLE CHOICE QUESTIONS ON LIMITS AND CONTINUITY\nBy: Rotimi\n\n---PAGE 1---\n\n1. Find the limit as x approaches 2:\nlim (x→2) (x² + 3x - 1)\nA. 7\nB. 8\nC. 9\nD. 10\n\n2. Determine if the function is continuous at x = 0:\nf(x) = {x² if x ≠ 0, 1 if x = 0}\nA. Continuous\nB. Not continuous\nC. Discontinuity removable\nD. Discontinuity jump\n\n[Continue with pages 2, 3, ... 18]"
  },
  "metadata": {
    "created_date": "2026-09-01",
    "ground_truth_type": "full_manual_transcription",
    "pages_transcribed": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
  }
}
```

### Documents to Populate (Priority Order)

1. **HIGHEST PRIORITY** (must have for baseline):
   - `DTS_SECOND_SEMESTER_EXAMINATION.json` (image, ~1-2 hours)
   - `IMG-20260627-WA0036.json` (image, ~0.5-1 hour)
   - `MTS_102_Multiple_Choice_Limits.json` (PDF, 18 pages, ~3-5 hours)

2. **MEDIUM PRIORITY** (recommended):
   - `MTS_102_Differentiation_Exercise.json` (PDF, 2 pages, ~1 hour)
   - `PHY_102_Practice_MCQ.json` (PDF, 21 pages, ~3-5 hours)

3. **LOW PRIORITY** (optional for baseline):
   - `MTS_102_Cheat_Sheet.json` (PDF, 4 pages, condensed content, ~1-2 hours)
   - `MTS102_Solutions.json` (PDF, mixed content, ~2-3 hours)

**Estimated total time for HIGHEST + MEDIUM**: ~12-18 hours of manual transcription

### Automation Considerations

For **faster ground truth creation**, consider:

1. **Use OCR tools for first pass** (e.g., cloud-based OCR APIs, Tesseract, etc.)
   - Run OCR on each document
   - Save output as initial ground truth
   - Manually review and correct errors
   - This is still faster than transcribing from scratch

2. **Crowdsourcing** (if available)
   - Distribute documents to reviewers
   - Each reviewer transcribes 1-2 documents
   - Aggregate and review for consistency

3. **Python Script to Validate**
   ```bash
   python -m app.core.documents.metrics
   # Runs validation on all ground truth files
   ```

---

## Phase 2: Benchmark Execution

### Prerequisites

Before running the benchmark, ensure:

1. **OpenVINO is installed**:
   ```bash
   pip install openvino>=2023.0
   ```

2. **PaddleOCR is installed**:
   ```bash
   pip install paddleocr>=2.7.0
   ```

3. **Dependencies are installed**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Models are in place**:
   ```
   models/ocr/openvino_models/
   ├── det/en_PP-OCRv3_det_infer.xml
   ├── det/en_PP-OCRv3_det_infer.bin
   ├── rec/ch_PP-OCRv3_rec_infer.xml
   └── rec/ch_PP-OCRv3_rec_infer.bin
   ```

5. **Ground truth is populated** (see Phase 1)

### Running the Benchmark

#### Basic Execution:

```bash
cd /path/to/ai-service

# Run benchmark on CPU (default)
python scripts/benchmark_ocr_baseline.py \
    --corpus-dir data/ocr_evaluation \
    --output results/baseline_ppocr_v3_cpu.json \
    --candidate ppocr_v3 \
    --device CPU
```

#### With GPU Acceleration:

```bash
python scripts/benchmark_ocr_baseline.py \
    --corpus-dir data/ocr_evaluation \
    --output results/baseline_ppocr_v3_gpu.json \
    --candidate ppocr_v3 \
    --device GPU
```

#### Advanced Options:

```bash
# Run with verbose logging
python -u scripts/benchmark_ocr_baseline.py \
    --corpus-dir data/ocr_evaluation \
    --output results/baseline_detailed.json \
    --candidate ppocr_v3 \
    --device CPU 2>&1 | tee results/baseline_execution.log

# Run subset of corpus (for testing)
# (Would need to modify script to add --limit option)
```

### Expected Output

The script produces:

1. **Console output**: Progress log with per-document status
2. **JSON results** (e.g., `results/baseline_ppocr_v3_cpu.json`):

```json
{
  "candidate": "ppocr_v3",
  "runtime": "openvino",
  "device": "CPU",
  "execution_date": "2026-09-15T14:32:00",
  "total_documents": 7,
  "successful": 7,
  "failed": 0,
  "avg_cer": 4.2,
  "avg_wer": 8.5,
  "median_cer": 3.8,
  "median_wer": 7.9,
  "total_latency_seconds": 245.3,
  "avg_latency_per_page_ms": 1203.5,
  "total_throughput_pages_per_sec": 0.83,
  "avg_peak_memory_mb": 512.4,
  "per_document_metrics": [
    {
      "filename": "DTS_SECOND_SEMESTER_EXAMINATION.jpeg",
      "document_type": "examination_paper",
      "page_count": 1,
      "character_error_rate": 3.2,
      "word_error_rate": 6.8,
      "total_latency_ms": 850.2,
      "latency_per_page_ms": 850.2,
      "pages_per_second": 1.18,
      "peak_memory_mb": 480.5,
      "success": true
    },
    ...
  ]
}
```

### Interpreting Results

#### CER (Character Error Rate)
- **0-2%**: Excellent (near-perfect transcription)
- **2-5%**: Good (acceptable for most use cases)
- **5-10%**: Fair (noticeable errors, manual review recommended)
- **>10%**: Poor (significant errors, model refinement needed)

#### WER (Word Error Rate)
- Typically 2-3× higher than CER
- **0-5%**: Excellent
- **5-15%**: Good
- **15-25%**: Fair
- **>25%**: Poor

#### Latency (ms/page)
- **<500ms**: Fast (excellent for real-time use)
- **500-1000ms**: Acceptable (suitable for batch processing)
- **1000-2000ms**: Slow (consider GPU acceleration)
- **>2000ms**: Very slow (requires optimization)

#### Memory (MB)
- Typical usage: 200-800 MB for PP-OCRv3 on CPU
- GPU variant: 1-4 GB
- If > 2 GB on CPU, consider device memory pressure

---

## Phase 3: Results Analysis

### Create Summary Report

After benchmark execution, create `results/BASELINE_ANALYSIS.md`:

```markdown
# PP-OCRv3 Baseline Benchmark Results

**Date**: 2026-09-15  
**Device**: CPU  
**Runtime**: OpenVINO  
**Candidate**: ppocr_v3  
**Total Documents**: 7  
**Success Rate**: 100% (7/7)

## Aggregate Metrics

| Metric | Value |
|--------|-------|
| Avg CER | 4.2% |
| Avg WER | 8.5% |
| Median CER | 3.8% |
| Median WER | 7.9% |
| Avg Latency/Page | 1203 ms |
| Throughput | 0.83 pages/sec |
| Avg Peak Memory | 512 MB |

## Per-Document Breakdown

### Images
- **DTS_SECOND_SEMESTER_EXAMINATION.jpeg**:
  - CER: 3.2% | WER: 6.8% | Latency: 850ms | Memory: 481 MB
  - Notes: High-quality scan, good performance

- **IMG-20260627-WA0036.jpg**:
  - CER: 5.1% | WER: 9.2% | Latency: 920ms | Memory: 495 MB
  - Notes: Mobile photo, perspective distortion, acceptable performance

### PDFs
- [Document analysis for each PDF]

## Assessment

- ✅ CER < 5%: Acceptable for BUILD's use case
- ✅ Performance: Suitable for batch processing
- ⚠️ Challenge: Mathematical notation (MTS documents)
- ⚠️ Challenge: Dense layouts (cheat sheet)

## Recommendations

1. Baseline is suitable for production
2. Consider GPU acceleration for throughput improvement
3. Investigate Surya alternative for mathematical text if needed
4. Proceed with multi-candidate benchmarking

## Next Steps

- [ ] Expand corpus to 50+ documents
- [ ] Test newer PaddleOCR version
- [ ] Benchmark Surya model
- [ ] Finalize model selection
- [ ] Implement winner in production pipeline
```

### Failure Analysis

If any documents fail to process:

1. **Check error logs** in `results/baseline_execution.log`
2. **Common errors**:
   - `OpenVINODeviceNotFoundError`: Device not available (use different device)
   - `OpenVINOModelNotFoundError`: Model files missing (verify paths)
   - `FileNotFoundError`: Document missing or inaccessible
   - `timeout`: Document too large (increase timeout)

3. **Create issue ticket** with:
   - Document name
   - Error type
   - Full error message
   - Log snippet

---

## Phase 4: Expand Corpus (Optional, if Baseline Succeeds)

If baseline metrics are acceptable (CER < 5%, success rate > 95%), expand corpus:

1. **Collect more representative documents**:
   - 10-20 additional exam papers
   - 5-10 documents with handwritten annotations
   - 5-10 documents with tables/complex layouts
   - Multi-lingual documents (if applicable)

2. **Repeat ground truth population** and benchmarking

3. **Aggregate results** across both datasets

---

## Phase 5: Multi-Candidate Comparison (After Baseline Succeeds)

Only proceed to multi-candidate benchmarking after baseline confirms:
- ✅ PP-OCRv3 CER < 5%
- ✅ Success rate > 95%
- ✅ Execution pipeline is stable

Then:
1. Acquire newer PaddleOCR version
2. Acquire Surya model
3. Run same benchmark harness on each candidate
4. Compare results using `data/ocr_evaluation/DECISION_FRAMEWORK.md`
5. Make evidence-based model selection

---

## Success Criteria for Baseline

Baseline is considered **successful** if:

✅ All 7 documents process without error  
✅ CER avg < 5%  
✅ WER avg < 10%  
✅ Latency < 2000 ms/page on CPU  
✅ Memory usage < 1 GB on CPU  
✅ Results reproducible on multiple runs  
✅ Ground truth fully populated  

If any criterion fails, investigate root cause and document findings.

---

## File Structure Reference

```
ai-service/
├── data/
│   └── ocr_evaluation/
│       ├── images/
│       │   ├── DTS_SECOND_SEMESTER_EXAMINATION.jpeg
│       │   └── IMG-20260627-WA0036.jpg
│       ├── pdfs/
│       │   ├── MTS_102_Multiple_Choice_Limits.pdf
│       │   ├── MTS_102_Differentiation_Exercise.pdf
│       │   ├── MTS_102_Cheat_Sheet.pdf
│       │   ├── MTS102_Solutions.pdf
│       │   └── PHY_102_Practice_MCQ.pdf
│       └── ground_truth/  ← Fill these in
│           ├── DTS_SECOND_SEMESTER_EXAMINATION.json
│           ├── IMG-20260627-WA0036.json
│           ├── MTS_102_Multiple_Choice_Limits.json
│           ├── MTS_102_Differentiation_Exercise.json
│           ├── MTS_102_Cheat_Sheet.json
│           ├── MTS102_Solutions.json
│           └── PHY_102_Practice_MCQ.json
├── scripts/
│   └── benchmark_ocr_baseline.py  ← Execution script
├── app/
│   └── core/
│       └── documents/
│           └── metrics.py  ← CER/WER calculation
└── results/
    ├── baseline_ppocr_v3_cpu.json  ← Output results
    ├── baseline_execution.log
    └── BASELINE_ANALYSIS.md
```

---

## Quick Reference: Commands

```bash
# 1. View corpus inventory
ls -lh data/ocr_evaluation/images/ data/ocr_evaluation/pdfs/

# 2. Check ground truth templates
ls -lh data/ocr_evaluation/ground_truth/

# 3. Run benchmark (requires OpenVINO + PaddleOCR)
python scripts/benchmark_ocr_baseline.py --corpus-dir data/ocr_evaluation

# 4. View results
cat results/baseline_ppocr_v3_cpu.json | python -m json.tool

# 5. Analyze metrics
python -c "
import json
with open('results/baseline_ppocr_v3_cpu.json') as f:
    result = json.load(f)
    print(f'CER: {result[\"avg_cer\"]:.2f}%')
    print(f'WER: {result[\"avg_wer\"]:.2f}%')
    print(f'Success rate: {result[\"successful\"]}/{result[\"total_documents\"]}')
"
```

---

## Contact & Escalation

- **Questions about ground truth**: Review existing JSON templates
- **Benchmark execution issues**: Check OpenVINO installation and model paths
- **Metrics interpretation**: Refer to "Interpreting Results" section
- **Model selection**: Follow decision framework in `DECISION_FRAMEWORK.md`

---

**Status**: Ready for ground truth population and baseline execution.  
**Owner**: BUILD AI/ML Team  
**Last Updated**: 2026-09-01
