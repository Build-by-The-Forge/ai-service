# OCR Model Candidates — Selection Framework

**Phase:** AI-2B.4.3 Decision Phase (to be filled in after benchmarking)

---

## Candidates Under Evaluation

| Candidate | Family | Status | OpenVINO | License | Document-Oriented | 
|-----------|--------|--------|----------|---------|-------------------|
| PP-OCRv3 | PaddleOCR | Baseline | ✅ | Apache-2.0 | Yes | 
| Newer PaddleOCR | PaddleOCR | Primary | ✅ | Apache-2.0 | Yes | 
| Surya | Surya | Alternative | ⚠️ | Non-commercial | Yes | 

---

## Evaluation Criteria

### 1. **Accuracy (Weight: 40%)**

Measured against BUILD's document types: ID cards, transcripts, certificates, notes, forms.

| Metric | PP-OCRv3 | Newer PaddleOCR | Surya | Winner |
|--------|----------|-----------------|-------|--------|
| Text Accuracy (WER %) | TBD | TBD | TBD | |
| Character Accuracy (CER %) | TBD | TBD | TBD | |
| Bounding Box F1 | TBD | TBD | TBD | |
| Table Extraction | TBD | TBD | TBD | |
| Math/Scientific Text | TBD | TBD | TBD | |
| Rotation/Skew Handling | TBD | TBD | TBD | |

**Subtotal Score:** ___/40

### 2. **Performance (Weight: 15%)**

Measured on standard CPU hardware (no GPU acceleration).

| Metric | PP-OCRv3 | Newer PaddleOCR | Surya | Winner |
|--------|----------|-----------------|-------|--------|
| Latency (ms/page) | TBD | TBD | TBD | |
| Memory Peak (MB) | TBD | TBD | TBD | |
| Model Size (MB) | TBD | TBD | TBD | |
| CPU Utilization | TBD | TBD | TBD | |

**Subtotal Score:** ___/15

### 3. **Deployment Fit (Weight: 20%)**

BUILD's target: Windows/Linux CPU-based inference via OpenVINO.

| Criterion | PP-OCRv3 | Newer PaddleOCR | Surya | Notes |
|-----------|----------|-----------------|-------|-------|
| OpenVINO Support | ✅ Native | ? | ? | Native or convertible? |
| Model Export Path | Clear | ? | ? | How to get production model? |
| Runtime Stability | ? | ? | ? | Any known issues? |
| Inference Framework | OpenVINO | ? | ? | Required dependencies? |

**Subtotal Score:** ___/20

### 4. **Language & Multilingual (Weight: 10%)**

BUILD is English-primary, but may expand to West African languages.

| Language | PP-OCRv3 | Newer PaddleOCR | Surya |
|----------|----------|-----------------|-------|
| English | ✅ | ✅ | ✅ |
| Multilingual Support | Limited | Broad | 90+ |
| Academic Text Quality | ? | ? | ? |

**Subtotal Score:** ___/10

### 5. **Licensing & Legal (Weight: 10%)**

BUILD cannot use non-commercial software in production.

| Aspect | PP-OCRv3 | Newer PaddleOCR | Surya |
|--------|----------|-----------------|-------|
| License | Apache-2.0 ✅ | Apache-2.0 ✅ | Non-commercial ⚠️ |
| Commercial Use | Allowed | Allowed | Waiver needed |
| Legal Risk | None | None | Moderate |
| Cost | Free | Free | Free (with waiver) |

**Subtotal Score:** ___/10

### 6. **Maintenance & Community (Weight: 5%)**

Long-term supportability and ecosystem health.

| Aspect | PP-OCRv3 | Newer PaddleOCR | Surya |
|--------|----------|-----------------|-------|
| Community Size | Large | Large | Growing |
| Update Frequency | Moderate | Frequent | Moderate |
| GitHub Activity | Active | Active | Active |
| Bus Factor | Medium | Medium | Medium |

**Subtotal Score:** ___/5

---

## Final Score Calculation

```
Score = (Accuracy × 0.40) 
       + (Performance × 0.15) 
       + (Deployment × 0.20) 
       + (Language × 0.10) 
       + (Licensing × 0.10) 
       + (Community × 0.05)

Maximum: 100 points
```

| Candidate | Total Score | Decision |
|-----------|------------|----------|
| PP-OCRv3 | TBD | |
| Newer PaddleOCR | TBD | |
| Surya | TBD | |

---

## Trade-offs Discussion

### Accuracy vs Performance

If Surya wins on accuracy but requires GPU or higher latency:
- **Decision:** Accuracy > performance for BUILD (education, correctness matters most)
- **Action:** Use Surya if accuracy gap > 5%, even at cost of latency

### OpenVINO Native vs Convertible

If Newer PaddleOCR is better overall but requires ONNX→OpenVINO conversion:
- **Decision:** Acceptable if conversion is stable and supported
- **Action:** Document conversion pipeline, add to build process

### Non-Commercial License (Surya)

If Surya is significantly better (10%+ accuracy improvement):
- **Decision:** Evaluate licensing waiver
- **Action:** Contact Surya team, review terms, assess legal risk
- **Fallback:** Use Newer PaddleOCR if waiver not acceptable

---

## Winner Selection Criteria

**Winning candidate must satisfy:**

1. ✅ **No legal blockers** (license acceptable for production)
2. ✅ **Reasonable performance** (< 5 sec/page on CPU)
3. ✅ **Deployable** (OpenVINO native or clear conversion path)
4. ✅ **Measurably better** than baseline (> 3% accuracy improvement)

---

## Post-Selection Actions

### Implementation Plan (if Newer PaddleOCR wins)

- [ ] Acquire official model artifacts
- [ ] Validate OpenVINO export
- [ ] Implement `NewerPaddleOCRProvider` adapter
- [ ] Add preprocessing (resize, normalize)
- [ ] Add postprocessing (bbox conversion)
- [ ] Unit tests matching contract
- [ ] Integration tests with full pipeline
- [ ] Performance benchmarks
- [ ] Documentation

### Rollback Plan (if unexpected issues)

- [ ] Keep PP-OCRv3 as fallback
- [ ] Implement runtime provider selection
- [ ] A/B test during ramp-up
- [ ] Monitor accuracy in production

---

## Notes

- This framework is filled in **after** benchmarking is complete
- Scores should be evidence-based (from benchmark results)
- Trade-offs must be explicitly documented
- Decision should be consensus among team

---

**Status:** READY FOR BENCHMARKING  
**Next Step:** Collect evaluation corpus and run benchmarks
