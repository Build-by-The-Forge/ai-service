# Phase 2 release gate R4 — model licences and evaluation material

**Status: CLOSED on 2026-10-14.** The owner signed off both sections (section 3): the OCR model licences (section 1) and the evaluation material (section 2). The gate had to close before the first production AI-service image could be published or deployed. Publishing still needs a separate, deliberate step: setting the repository variable `PUBLISH_IMAGES` to `true` (`.github/workflows/backend.yml`).
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
- the attribution "PaddleOCR, Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved." ships with the models.

**Open on 2026-10-11:** the redistribution terms of the pretrained weights themselves. A software licence on the package does not by itself cover every model artifact.

### Review, 2026-10-14

A technical licence review, not a legal opinion. All sources were accessed on 2026-10-14.

| Source | What it says |
|---|---|
| [PaddleOCR issue #8780](https://github.com/PaddlePaddle/PaddleOCR/issues/8780) | Asked (2023-01-04) about the licence of `ch_PP-OCRv3_det`, `en_PP-OCRv3_rec`, `en_PP-OCRv3_rec_slim` and `ch_ppocr_mobile_v2.0_cls`. A PaddleOCR collaborator (tink2123, also the assignee) replied on 2023-01-05: "They all under http://www.apache.org/licenses/LICENSE-2.0". |
| [PaddleOCR `release/2.7` LICENSE](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/LICENSE) | Apache License 2.0, "Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved". Byte-identical to the `LICENSE` in the `paddleocr` 2.7.0 wheel the image installs (SHA-256 `ff6311c56c1b22c9209c96cc8afc16da91c24ee55a15e14ebc46fe1960dbd856`). |
| [PaddleOCR `release/2.7` model list](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/models_list_en.md) | Publishes all three models with exactly the download URLs above. States no licence of its own. |
| [PaddlePaddle/PP-OCRv3_mobile_det](https://huggingface.co/PaddlePaddle/PP-OCRv3_mobile_det) | PaddlePaddle's card for the PP-OCRv3 detection series: `apache-2.0`. Does not name `en_PP-OCRv3_det`. |
| [PaddlePaddle/en_PP-OCRv3_mobile_rec](https://huggingface.co/PaddlePaddle/en_PP-OCRv3_mobile_rec) | PaddlePaddle's card for the PP-OCRv3 English recognition model: `apache-2.0`. |
| [OpenCV Zoo, `text_detection_ppocr`](https://huggingface.co/opencv/opencv_zoo/blob/main/models/text_detection_ppocr/README.md) | Third party, corroboration only. Cites `en_PP-OCRv3_det_infer.tar` as its source model and licenses the directory under Apache 2.0. |

**Per archive:**
- `en_PP-OCRv3_rec_infer`: Apache 2.0, **stated** upstream (issue #8780). PaddlePaddle's card for the same model family agrees.
- `ch_ppocr_mobile_v2.0_cls_infer`: Apache 2.0, **stated** upstream (issue #8780).
- `en_PP-OCRv3_det_infer`: Apache 2.0, **inferred**. No upstream statement names this model. It is published by the Apache-2.0 PaddleOCR project in its official model list. Its Chinese sibling `ch_PP-OCRv3_det` is covered by issue #8780, and PaddlePaddle's card for the PP-OCRv3 detection series is Apache-2.0. No source found suggests other terms. Whoever signs off accepts this inference.

**Obligations (Apache 2.0, section 4), and how they are met:**
- (a) A copy of the licence goes to recipients. In the image it is `paddleocr-*.dist-info/LICENSE`. With the release it is `models/ocr/LICENSE`, the same file, attached next to the archives.
- (b) Modified files carry change notices. Not applicable: the archives are redistributed unmodified (SHA-256 verified).
- (c), (d) Retain attribution notices and any upstream NOTICE file. The attribution is in `models/ocr/NOTICE.md`, which the image installs at `$OCR_MODEL_DIR/NOTICE.md` and the release also carries. PaddleOCR publishes no NOTICE file.

**Done:**
1. Licence statements found and recorded above, per archive.
2. `models/ocr/NOTICE.md` updated: attribution, licence, where the full text is, and the evidence. `models/ocr/LICENSE` added.
3. The manifest's `license` fields changed to `Apache-2.0`.

The `ocr-models-2026.10.0` release must carry exactly the three archives (each matching the manifest's SHA-256), `NOTICE.md` and `LICENSE`. Nothing from `data/ocr_evaluation`, no evaluation results and no other files go into it. `scripts/fetch_ocr_models.py` finds each archive by name, so the two text assets do not affect it.

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

### Owner decision, 2026-10-14

The owner (AI-MLGuru) confirms that this material is meant for the public and is cleared for public distribution, including in this repository. The owner made this decision after seeing the privacy review below, including the phone-number-like string in `MTS102_Solutions.pdf`. No file needs removing or replacing (step 2).

**Privacy review (step 3 and every other file), 2026-10-14:**
- Both images were inspected visually.
- Image metadata was checked for EXIF and GPS.
- PDF metadata and text were scanned for email addresses, phone numbers, matriculation or registration numbers, and URLs.
- Ground-truth transcriptions were scanned for the same patterns.
- Only kinds and counts were recorded, not values.

| File | Provenance | Basis for distribution | Privacy review |
|---|---|---|---|
| `pdfs/MTS 102  Multiple Choice Questions on Limits and Continuity BY ROTIMI.pdf` | Course material; author named in the file name | Owner confirmation: meant for the public | 18 pages with almost no text layer (scanned). Nothing found in the extractable text; the page images were not read. No author metadata |
| `pdfs/MTS 102 - Differentiation Exercise  BY PHOTOCHEM.pdf` | Course material; author or handle in the file name | Owner confirmation: meant for the public | Nothing found. No author metadata |
| `pdfs/MTS 102 Cheat sheet.pdf_20250907_152115_0000.pdf` | Not recorded beyond the owner's confirmation | Owner confirmation: meant for the public | Nothing found in the text. The PDF author field holds a two-word personal name |
| `pdfs/MTS102_Solutions.pdf` | Not recorded beyond the owner's confirmation | Owner confirmation: meant for the public | Page 16 ends with a sign-off, "Emerald", followed by an 11-digit number shaped like a mobile number. The owner confirms it is part of the published material |
| `pdfs/PHY 102 PRACTICE MULTIPLE CHOICE QUESTIONS BY GRACE..pdf` | Course material; author named in the file name | Owner confirmation: meant for the public | Nothing found in the text. The PDF author field holds a single name |
| `images/DTS SECOND SEMESTER 20252026 EXAMINATION.jpeg` | Federal University of Technology, Akure: DTS 104 examination paper, April 2025 | Owner confirmation: meant for the public | No personal data visible. No EXIF or GPS metadata |
| `images/IMG-20260627-WA0036.jpg` | Messaging-app photo of a handwritten practical worksheet (boiling time against salt) on WAEC/NECO graph paper | Owner confirmation: meant for the public | No names, faces, phone numbers or ID numbers visible. No EXIF or GPS metadata |
| `ground_truth/*.json` (7 files) | Our own transcriptions of the documents above | Follows the source documents | Nothing found |

## 3. Sign-off

| Item | Confirmed by | Date | Source / basis |
|---|---|---|---|
| `en_PP-OCRv3_det_infer` redistribution | AI-MLGuru (owner) | 2026-10-14 | Apache-2.0, inferred: PaddleOCR LICENSE, official model list, PP-OCRv3 detection-series card, issue #8780 (sibling `ch_PP-OCRv3_det`). The owner accepts this documented inference and its limitation: no upstream statement names this exact archive. Section 1 |
| `en_PP-OCRv3_rec_infer` redistribution | AI-MLGuru (owner) | 2026-10-14 | Apache-2.0, stated: issue #8780; section 1 |
| `ch_ppocr_mobile_v2.0_cls_infer` redistribution | AI-MLGuru (owner) | 2026-10-14 | Apache-2.0, stated: issue #8780; section 1 |
| `data/ocr_evaluation` files (each) | AI-MLGuru (owner) | 2026-10-14 | Owner confirms each file is meant for the public and cleared for public distribution; per-file provenance and privacy review in section 2 |
