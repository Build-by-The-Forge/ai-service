# Third-party OCR models: notices

The AI-service image contains three pretrained inference models from the PaddleOCR project, installed by `scripts/fetch_ocr_models.py` from `models/ocr/manifest.json` and verified against the SHA-256 recorded there:

| Model | Role | Upstream archive |
|---|---|---|
| `en_PP-OCRv3_det_infer` | text detection | https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar |
| `en_PP-OCRv3_rec_infer` | English text recognition | https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar |
| `ch_ppocr_mobile_v2.0_cls_infer` | text-angle classification | https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar |

**Attribution:** PaddleOCR, Copyright (c) PaddlePaddle Authors. https://github.com/PaddlePaddle/PaddleOCR

**Software licence:** the `paddleocr` and `paddlepaddle` packages are released under the Apache License 2.0. The full text ships with each package in the image (its `*.dist-info/LICENSE`).

## Release gate R4: not yet confirmed

The package licence does not by itself establish the redistribution terms of each pretrained model. Before the first production image, someone accountable for releases must:

1. Confirm the licence and any attribution or notice requirements that apply to each of the three exact archives above. Use the model's release page or the model-zoo documentation of the PaddleOCR version that published it, and record the source in this file.
2. Add any required notice text here. This file is copied into the image next to the models, at `$OCR_MODEL_DIR/NOTICE.md`.

Until both are done, the manifest marks each model's redistribution terms as *pending confirmation (R4)*.
