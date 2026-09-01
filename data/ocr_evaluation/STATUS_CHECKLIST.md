# OCR Baseline Benchmarking: Setup Phase Completion Checklist

**Created**: 2026-09-01  
**Phase**: Infrastructure Setup Complete ✅  
**Next Phase**: Ground Truth Population & Baseline Execution

---

## ✅ Completed Infrastructure

### Ground Truth Templates
- [x] `DTS_SECOND_SEMESTER_EXAMINATION.json` — Image template
- [x] `IMG-20260627-WA0036.json` — Image template  
- [x] `MTS_102_Multiple_Choice_Limits.json` — 18-page PDF template
- [x] `MTS_102_Differentiation_Exercise.json` — 2-page PDF template
- [x] `MTS_102_Cheat_Sheet.json` — 4-page PDF template
- [x] `MTS102_Solutions.json` — Multi-page PDF template
- [x] `PHY_102_Practice_MCQ.json` — 21-page PDF template

All templates include:
- [x] Document metadata (type, source, page count, quality notes)
- [x] Ground truth structure (full_text + text_regions)
- [x] Expected challenges documentation
- [x] Standard JSON format

### Benchmark Harness
- [x] `scripts/benchmark_ocr_baseline.py` created
- [x] Async/await implementation for efficient processing
- [x] Per-document metrics collection
- [x] Aggregate statistics calculation
- [x] JSON export functionality
- [x] Error handling and categorization
- [x] Logging infrastructure
- [x] OpenVINO device discovery support
- [x] Extensible for multi-candidate comparison

### Metrics Calculation
- [x] `app/core/documents/metrics.py` implemented
- [x] Levenshtein distance algorithm
- [x] CER (Character Error Rate) calculation
- [x] WER (Word Error Rate) calculation
- [x] EditDistanceResult dataclass
- [x] GroundTruthLoader for JSON parsing
- [x] ComparisonResult dataclass
- [x] compare_ocr_to_ground_truth() function

### Testing & Validation
- [x] Metrics module unit tests (15+ assertions)
- [x] CER tests: perfect match, single/multiple errors, edge cases
- [x] WER tests: word-level error detection
- [x] Multi-line text tests
- [x] Empty string edge cases
- [x] OCR-realistic scenarios

### Documentation
- [x] `BASELINE_EXECUTION_GUIDE.md` — 4-phase execution guide
  - [x] Phase 1: Ground Truth Population (step-by-step instructions)
  - [x] Phase 2: Benchmark Execution (prerequisites, commands)
  - [x] Phase 3: Results Analysis (templates, troubleshooting)
  - [x] Phase 4: Corpus Expansion (optional next steps)

- [x] `docs/AI-2B-4-3-BASELINE-SETUP-COMPLETE.md` — Summary documentation
  - [x] Architecture overview
  - [x] File structure reference
  - [x] Success criteria
  - [x] Metrics interpretation guide
  - [x] Troubleshooting reference

---

## ⏳ Blocking: Ground Truth Population

### Status: NOT STARTED (Required before execution)

### Documents to Populate (Priority Order)

#### Priority 1: CRITICAL FOR BASELINE (7-10 hours)
- [ ] `data/ocr_evaluation/ground_truth/DTS_SECOND_SEMESTER_EXAMINATION.json`
  - Source: `data/ocr_evaluation/images/DTS SECOND SEMESTER 20252026 EXAMINATION.jpeg`
  - Type: Image (1 page, 905x1280)
  - Estimated time: 1-2 hours
  - Challenge: Exam paper with varied text sizes

- [ ] `data/ocr_evaluation/ground_truth/IMG-20260627-WA0036.json`
  - Source: `data/ocr_evaluation/images/IMG-20260627-WA0036.jpg`
  - Type: Image (1 page, 960x1280)
  - Estimated time: 0.5-1 hour
  - Challenge: Mobile photo with perspective distortion

- [ ] `data/ocr_evaluation/ground_truth/MTS_102_Multiple_Choice_Limits.json`
  - Source: `data/ocr_evaluation/pdfs/MTS 102  Multiple Choice Questions on Limits and Continuity BY ROTIMI.pdf`
  - Type: PDF (18 pages, 4.5 MB)
  - Estimated time: 3-5 hours
  - Challenge: Multi-page, mathematical content

#### Priority 2: RECOMMENDED (4-6 hours)
- [ ] `data/ocr_evaluation/ground_truth/MTS_102_Differentiation_Exercise.json`
  - Source: `data/ocr_evaluation/pdfs/MTS 102 - Differentiation Exercise  BY PHOTOCHEM.pdf`
  - Type: PDF (2 pages, 43 KB)
  - Estimated time: 1 hour
  - Challenge: Mathematical notation

- [ ] `data/ocr_evaluation/ground_truth/PHY_102_Practice_MCQ.json`
  - Source: `data/ocr_evaluation/pdfs/PHY 102 PRACTICE MULTIPLE CHOICE QUESTIONS BY GRACE..pdf`
  - Type: PDF (21 pages, 647 KB)
  - Estimated time: 3-5 hours
  - Challenge: Large document, physics notation

#### Priority 3: OPTIONAL (3-5 hours)
- [ ] `data/ocr_evaluation/ground_truth/MTS_102_Cheat_Sheet.json`
  - Source: `data/ocr_evaluation/pdfs/MTS 102 Cheat sheet.pdf_20250907_152115_0000.pdf`
  - Type: PDF (4 pages, 135 KB)
  - Estimated time: 1-2 hours
  - Challenge: Dense content, multiple columns

- [ ] `data/ocr_evaluation/ground_truth/MTS102_Solutions.json`
  - Source: `data/ocr_evaluation/pdfs/MTS102_Solutions.pdf`
  - Type: PDF (? pages, 247 KB)
  - Estimated time: 2-3 hours
  - Challenge: Mixed content type

### How to Populate

For each JSON file:

1. **Open the source document**
   ```bash
   # Image: Use image viewer
   open data/ocr_evaluation/images/DTS*.jpeg
   
   # PDF: Use PDF reader or render to images
   open data/ocr_evaluation/pdfs/MTS*.pdf
   ```

2. **Read the template JSON**
   ```bash
   cat data/ocr_evaluation/ground_truth/DTS_SECOND_SEMESTER_EXAMINATION.json
   ```

3. **Manually transcribe text**
   - All readable text exactly as shown
   - Preserve structure (headings, questions, options)
   - Note illegible content as `[illegible]`
   - Note diagrams/charts as `[diagram]`, `[chart]`

4. **Fill `ground_truth.full_text` field**
   ```json
   {
     "ground_truth": {
       "full_text": "[Your transcription here]",
       "text_regions": [...]
     }
   }
   ```

5. **Validate JSON**
   ```bash
   python -m json.tool data/ocr_evaluation/ground_truth/DTS_SECOND_SEMESTER_EXAMINATION.json > /dev/null && echo "✓ Valid JSON"
   ```

6. **Save file**

### Automation Option

If manual transcription is too time-consuming:

1. Use cloud OCR API (AWS Textract, Google Vision, Azure Vision) for first pass
2. Save output as initial ground truth
3. Manually review and correct errors
4. Faster than transcribing from scratch

---

## ⏳ Next Steps (After Ground Truth Population)

### 1. Baseline Execution
```bash
# Ensure OpenVINO and PaddleOCR installed
pip install openvino>=2023.0 paddleocr>=2.7.0

# Run baseline benchmark
python scripts/benchmark_ocr_baseline.py \
    --corpus-dir data/ocr_evaluation \
    --output results/baseline_ppocr_v3_cpu.json \
    --candidate ppocr_v3 \
    --device CPU
```

**Expected time**: 5-30 minutes depending on document sizes

**Output**: `results/baseline_ppocr_v3_cpu.json` with metrics

### 2. Results Analysis
- [ ] Check success rate (target: 100%)
- [ ] Verify CER (target: < 5%)
- [ ] Verify WER (target: < 10%)
- [ ] Check latency (target: < 2000 ms/page on CPU)
- [ ] Check memory (target: < 1 GB on CPU)

### 3. Create Summary Report
```bash
# Review metrics and create results/BASELINE_ANALYSIS.md
# Document findings, recommendations, and next actions
```

### 4. Proceed to Multi-Candidate (Only If Baseline Succeeds)
- [ ] Expand corpus (optional)
- [ ] Acquire newer PaddleOCR version
- [ ] Acquire Surya model
- [ ] Run same benchmark on each candidate
- [ ] Compare results using DECISION_FRAMEWORK.md

---

## Success Criteria Checklist

### Infrastructure Setup (This Session)
- [x] Ground truth templates created
- [x] Benchmark harness script ready
- [x] Metrics calculation module tested
- [x] Execution documentation complete
- [x] Troubleshooting reference available

### Ground Truth Population (Next Step)
- [ ] All 7 documents have `full_text` field populated
- [ ] JSON files are valid (parseable)
- [ ] Text is accurately transcribed (no fabrication)
- [ ] Estimated time: 12-18 hours

### Baseline Execution (After Ground Truth)
- [ ] All 7 documents process without error (100% success)
- [ ] CER average < 5%
- [ ] WER average < 10%
- [ ] Latency < 2000 ms/page on CPU
- [ ] Memory < 1 GB on CPU
- [ ] Results reproducible on multiple runs
- [ ] Estimated time: 5-30 minutes

### Baseline Analysis (After Execution)
- [ ] Summary report created
- [ ] Findings documented
- [ ] Recommendations provided
- [ ] Blockers identified (if any)
- [ ] Estimated time: 1-2 hours

---

## Quick Start Reference

### Check Corpus Inventory
```bash
ls -lh data/ocr_evaluation/images/
ls -lh data/ocr_evaluation/pdfs/
```

### View Ground Truth Templates
```bash
ls -lh data/ocr_evaluation/ground_truth/
cat data/ocr_evaluation/ground_truth/DTS_SECOND_SEMESTER_EXAMINATION.json | python -m json.tool | head -30
```

### Test Metrics Module (No Dependencies)
```bash
python app/core/documents/metrics.py
```

### Validate JSON Ground Truth
```bash
for f in data/ocr_evaluation/ground_truth/*.json; do
  python -m json.tool "$f" > /dev/null && echo "✓ $f" || echo "✗ $f"
done
```

### Run Benchmark (Requires OpenVINO + PaddleOCR)
```bash
python scripts/benchmark_ocr_baseline.py \
    --corpus-dir data/ocr_evaluation \
    --output results/baseline_ppocr_v3_cpu.json
```

---

## File Locations Summary

| What | Where |
|------|-------|
| Documents to OCR | `data/ocr_evaluation/images/` + `pdfs/` |
| Ground truth to populate | `data/ocr_evaluation/ground_truth/*.json` |
| Benchmark script | `scripts/benchmark_ocr_baseline.py` |
| Metrics module | `app/core/documents/metrics.py` |
| Execution guide | `data/ocr_evaluation/BASELINE_EXECUTION_GUIDE.md` |
| Setup summary | `docs/AI-2B-4-3-BASELINE-SETUP-COMPLETE.md` |
| Results (will be created) | `results/baseline_ppocr_v3_cpu.json` |
| Results log (will be created) | `results/baseline_execution.log` |

---

## Timeline

| Phase | Task | Time | Status |
|-------|------|------|--------|
| **SETUP** | Infrastructure & templates | ✅ Complete | 2-3 hours |
| **DATA** | Ground truth population | ⏳ Next | 12-18 hours |
| **EXEC** | Baseline benchmark run | ⏳ Then | 5-30 min |
| **ANALYSIS** | Results review & report | ⏳ Then | 1-2 hours |
| **EXPAND** | Multi-candidate comparison | ❌ Deferred | 20-40 hours |

**Total for Baseline**: ~20-26 hours (mostly manual transcription)

---

## Contact & Support

- **Question about templates?** → See `BASELINE_EXECUTION_GUIDE.md` Phase 1
- **Question about execution?** → See `BASELINE_EXECUTION_GUIDE.md` Phase 2
- **Question about metrics?** → Run `python app/core/documents/metrics.py` for examples
- **Troubleshooting?** → See `BASELINE_SETUP-COMPLETE.md` "Troubleshooting Reference"

---

**Status**: Ready for ground truth population  
**Owner**: BUILD AI/ML Team  
**Created**: 2026-09-01  
**Last Updated**: This session

---

## Next Immediate Action

👉 **START HERE**: Open `data/ocr_evaluation/ground_truth/DTS_SECOND_SEMESTER_EXAMINATION.json` and populate the `ground_truth.full_text` field with the transcription from the exam image.

Estimated time: 1-2 hours for this one document.

After completing the 2 images + 1 large PDF (Priority 1), you'll be ready to execute the baseline benchmark.
