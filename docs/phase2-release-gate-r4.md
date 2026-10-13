# Phase 2 release gate R4 — model licences and evaluation material

**Status: OPEN.** Must be closed before the first production AI-service image is published or deployed.
**Recorded:** 2026-10-11, during Phase 2 Step 2.

## 1. OCR model redistribution

The image ships three pretrained PaddleOCR models (`models/ocr/manifest.json`, `models/ocr/NOTICE.md`):

| Model | Archive SHA-256 | Upstream archive |
|---|---|---|
| `en_PP-OCRv3_det_infer` | `e740eb52…8794f` | `https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar` |
| `en_PP-OCRv3_rec_infer` | `94402088…23343` | `https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar` |
| `ch_ppocr_mobile_v2.0_cls_infer` | `50735258…57717` | `https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar` |

**Established:**
- the `paddleocr` 2.7.0 package declares Apache License 2.0 (package metadata);
- the attribution "PaddleOCR, Copyright (c) PaddlePaddle Authors" ships with the models.

**Not established:** the redistribution terms of the pretrained weights themselves. A software licence on the package does not by itself cover every model artifact.

**To close:**
1. For each archive above, find the licence statement that covers it (model-zoo page, release notes, or the licence file of the PaddleOCR version that published it). Record the source URL and date here.
2. Copy any required notice text into `models/ocr/NOTICE.md`, which the image installs at `$OCR_MODEL_DIR/NOTICE.md`.
3. Change the manifest's `license` fields from "pending confirmation (R4)" to the confirmed terms.

## 2. Evaluation material in `data/ocr_evaluation`

These files were committed on 2026-09-01 (`43aeb56`) for the AI-2B.4.3 OCR benchmark:

| File | Size | Notes |
|---|---|---|
| `pdfs/MTS 102  Multiple Choice Questions on Limits and Continuity BY ROTIMI.pdf` | 4.7 MB | Third-party course material; author named in the file name |
| `pdfs/MTS 102 - Differentiation Exercise  BY PHOTOCHEM.pdf` | 43 KB | Third-party; author or handle in the file name |
| `pdfs/MTS 102 Cheat sheet.pdf_20250907_152115_0000.pdf` | 137 KB | Provenance unknown |
| `pdfs/MTS102_Solutions.pdf` | 252 KB | Provenance unknown |
| `pdfs/PHY 102 PRACTICE MULTIPLE CHOICE QUESTIONS BY GRACE..pdf` | 662 KB | Third-party; author named in the file name |
| `images/DTS SECOND SEMESTER 20252026 EXAMINATION.jpeg` | 131 KB | An institution's examination paper |
| `images/IMG-20260627-WA0036.jpg` | 120 KB | A messaging-app photo; **may contain personal data** |
| `ground_truth/*.json` | small | Our own transcriptions of the documents above |

**Already done (Step 2):**
- None of this material reaches the AI-service image. The Dockerfile copies only what the service runs, and `data/` is not part of it.
- No test or application code reads `data/ocr_evaluation`; only the benchmark scripts do, and only by explicit argument.

**Not established:** the right to keep or redistribute these files, including in the repository itself, which is distributable to anyone given access.

**To close, per file:**
1. Record its provenance and the legal basis for holding it (licence, the author's permission, or another basis).
2. Where that basis is unclear, remove the file from the repository (history needs a separate decision). Replace it with a permitted fixture, such as a document we author, a public-domain text or a properly licensed sample, so that benchmark coverage of each layout (native PDF, scanned exam, phone photo) survives.
3. Check the messaging-app photo specifically for personal data before any further use.

## 3. Sign-off

| Item | Confirmed by | Date | Source / basis |
|---|---|---|---|
| `en_PP-OCRv3_det_infer` redistribution | | | |
| `en_PP-OCRv3_rec_infer` redistribution | | | |
| `ch_ppocr_mobile_v2.0_cls_infer` redistribution | | | |
| `data/ocr_evaluation` files (each) | | | |
