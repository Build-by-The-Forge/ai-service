# OCR Model Evaluation — Benchmarking Checklist

**Phase:** AI-2B.4.3
**Status:** Planning phase
**Target:** Evidence-based selection of production OCR stack

---

## Part 1: Baseline Preparation

### Baseline: PP-OCRv3/OpenVINO

- [x] Already have IDQORE models in repo
  - `models/ocr/openvino_models/det/en_PP-OCRv3_det_infer.{xml,bin}`
  - `models/ocr/openvino_models/rec/ch_PP-OCRv3_rec_infer.{xml,bin}`
- [ ] Tag as `ppocr_v3_baseline` in version tracking
- [ ] Verify both detection + recognition model contracts
- [ ] Document expected performance baseline

---

## Part 2: Acquire Candidates

### Candidate 1: Newer PaddleOCR

- [ ] Research latest PaddleOCR releases (v3+)
- [ ] Confirm OpenVINO export support
- [ ] Download model artifacts (det + rec)
- [ ] Verify input/output contract (shapes, dtypes)
- [ ] Create test harness for newer PaddleOCR
- [ ] Location: `models/ocr/openvino_models/paddleocr_new/`

### Candidate 2: Surya

- [ ] Research Surya OCR architecture
- [ ] Check licensing (non-commercial waiver process)
- [ ] Determine OpenVINO deployment path (or ONNX, TensorRT)
- [ ] Download model artifacts
- [ ] Create test harness for Surya
- [ ] Location: `models/ocr/openvino_models/surya/`

---

## Part 3: Build Evaluation Corpus

### Image Test Set

- [ ] Scan or collect:
  - [ ] 5-10 ID cards (various formats)
  - [ ] 5-10 passport pages
  - [ ] 5-10 academic certificates
  - [ ] 5-10 academic transcripts
  - [ ] 10-15 lecture notes (handwritten + typed)
  - [ ] 5-10 forms with tables
  - [ ] 5-10 pages with mathematical/scientific text
  - [ ] 10-15 low-quality/compressed images
  - [ ] 10-15 rotated/skewed pages
- [ ] Store in `data/ocr_evaluation/images/`
- [ ] Document source and anonymization

### PDF Test Set

- [ ] Scan or collect:
  - [ ] 5-10 academic transcripts (multi-page, native text)
  - [ ] 5-10 fully scanned PDFs (no native text)
  - [ ] 5-10 mixed PDFs (native + scanned)
  - [ ] 5-10 forms with tables
  - [ ] 5-10 multi-column documents
  - [ ] 5-10 PDFs with rotated pages
- [ ] Store in `data/ocr_evaluation/pdfs/`
- [ ] Verify PDFs are anonymized (FERPA/GDPR safe)

### Ground Truth Annotations

- [ ] For each image/PDF, create JSON ground truth:
  - [ ] Expected OCR text
  - [ ] Bounding box regions
  - [ ] Confidence expectations
  - [ ] Document type metadata
  - [ ] Language labels
  - [ ] Content complexity notes
- [ ] Store in `data/ocr_evaluation/ground_truth/`
- [ ] Format: `{filename}.json`

---

## Part 4: Benchmark Infrastructure

### Benchmarking Scripts

- [ ] Create `scripts/benchmark_ocr.py`
  - [ ] Load candidate model
  - [ ] Process all images in corpus
  - [ ] Process all PDFs in corpus
  - [ ] Run inference on CPU (build target)
  - [ ] Capture latency, memory, output
  - [ ] Compare against ground truth
  - [ ] Generate per-candidate results JSON

- [ ] Create `scripts/compare_ocr_results.py`
  - [ ] Load all candidate results
  - [ ] Compute metrics (WER, CER, detection F1, etc.)
  - [ ] Generate comparison table
  - [ ] Output HTML report
  - [ ] Highlight strengths/weaknesses

### Measurement Categories

- [ ] **Text Accuracy:** WER, CER per document type
- [ ] **Detection Quality:** Precision, recall, F1 on bounding boxes
- [ ] **Layout Preservation:** Order correctness, spacing
- [ ] **Structured Data:** Table extraction accuracy
- [ ] **Special Cases:** Math text, rotation, low-quality
- [ ] **Performance:** Latency (ms/page), memory peak (MB)
- [ ] **Multilingual:** Language support (English, others if relevant)
- [ ] **Licensing:** Production readiness, restrictions

---

## Part 5: Run Benchmarks

- [ ] Set up isolated test environment
- [ ] Run PP-OCRv3 baseline
  - [ ] Record: accuracy, latency, memory
  - [ ] Save output: `results/ppocr_v3.json`
  - [ ] Flag: any model contract issues
- [ ] Run Newer PaddleOCR
  - [ ] Record: accuracy, latency, memory
  - [ ] Save output: `results/paddleocr_new.json`
  - [ ] Flag: any deployment issues
- [ ] Run Surya
  - [ ] Record: accuracy, latency, memory
  - [ ] Save output: `results/surya.json`
  - [ ] Flag: licensing implications
- [ ] Generate comparison report
  - [ ] `results/comparison.html`
  - [ ] Tabular metrics
  - [ ] Visual winners per category

---

## Part 6: Decision

- [ ] Review benchmark results
- [ ] Evaluate trade-offs:
  - Accuracy vs speed vs memory vs licensing
  - BUILD's priority: document accuracy > performance > cost
- [ ] Select primary candidate
- [ ] Document decision in `data/ocr_evaluation/DECISION.md`
  - [ ] Rationale
  - [ ] Trade-offs accepted
  - [ ] Next steps (implementation plan)

---

## Part 7: Implementation

- [ ] Only after benchmark decision is made
- [ ] Create model-specific adapter in `app/core/documents/`
- [ ] Add preprocessing (resize, normalize, etc.)
- [ ] Add postprocessing (bounding box conversion, etc.)
- [ ] Integrate behind existing `OCRProvider` boundary
- [ ] Write unit tests
- [ ] Verify contracts unchanged

---

## Success Criteria

✅ **Evidence-based decision** — benchmark data drives the choice, not assumptions
✅ **Reproducible** — anyone can re-run benchmarks with the same corpus
✅ **Fair comparison** — all candidates tested on identical inputs
✅ **Documented** — rationale and trade-offs clearly recorded
✅ **BUILD-focused** — test cases represent real university documents
✅ **Low regret** — choice is clearly better than alternatives for BUILD's use case

---

## Timeline Estimate

- Baseline validation: 1-2 days
- Acquire candidates: 3-5 days (model search, validation)
- Build evaluation corpus: 5-10 days (data collection, annotation)
- Benchmark infrastructure: 3-5 days (script development)
- Run benchmarks: 2-3 days (execution, analysis)
- Decision + documentation: 1 day
- **Total:** 15-26 days of focused work

---

See `data/ocr_evaluation/README.md` for dataset structure.
