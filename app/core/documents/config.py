from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class OCRConfig:
    """
    Configuration contract governing the OpenVINO / PaddleOCR inference runtime.
    """

    detection_input_size: int = 960
    detection_threshold: float = 0.3
    box_threshold: float = 0.6
    unclip_ratio: float = 1.5
    recognition_batch_size: int = 8
    recognition_threshold: float = 0.5
    max_pages: int = 100
    max_image_pixels: int = 16_000_000
    device: str = "CPU"
    model_dir: str = "models/ocr"
    det_model_name: str = "ch_PP-OCRv3_det_infer"
    rec_model_name: str = "ch_PP-OCRv3_rec_infer"
    cls_model_name: str = "ch_ppocr_mobile_v2.0_cls_infer"
    enable_orientation_classification: bool = True
    enable_table_extraction: bool = True
    quality_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "min_blur_score": 100.0,
            "min_contrast": 20.0,
            "min_confidence": 0.5,
        }
    )
