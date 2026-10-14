# Third-party OCR models: notices

The AI-service image contains three pretrained inference models from the PaddleOCR project, installed by `scripts/fetch_ocr_models.py` from `models/ocr/manifest.json` and verified against the SHA-256 recorded there:

| Model | Role | Upstream archive |
|---|---|---|
| `en_PP-OCRv3_det_infer` | text detection | https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar |
| `en_PP-OCRv3_rec_infer` | English text recognition | https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar |
| `ch_ppocr_mobile_v2.0_cls_infer` | text-angle classification | https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar |

**Attribution:** PaddleOCR, Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved. https://github.com/PaddlePaddle/PaddleOCR

**Licence:** the three models are distributed under the Apache License, Version 2.0. They are redistributed unmodified: each archive matches the upstream archive byte for byte (same SHA-256), and the image only extracts it.

**Licence text:** `models/ocr/LICENSE` in the repository, attached to the `ocr-models-2026.10.0` release next to the archives. It is PaddleOCR's own licence file, byte-identical to the one in the `paddleocr` 2.7.0 package, which the image ships as `paddleocr-*.dist-info/LICENSE`. PaddleOCR publishes no NOTICE file, so there is no upstream notice text to reproduce.

## Licence evidence (release gate R4)

Recorded in `docs/phase2-release-gate-r4.md`, accessed 2026-10-14:

- `en_PP-OCRv3_rec_infer`, `ch_ppocr_mobile_v2.0_cls_infer`: a PaddleOCR collaborator states the listed pretrained models are under Apache 2.0, https://github.com/PaddlePaddle/PaddleOCR/issues/8780
- `en_PP-OCRv3_det_infer`: no upstream statement names this model. Apache 2.0 is inferred from the PaddleOCR licence (https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/LICENSE), the official model list that publishes it (https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/models_list_en.md), and PaddlePaddle's Apache-2.0 card for the PP-OCRv3 detection series (https://huggingface.co/PaddlePaddle/PP-OCRv3_mobile_det).
