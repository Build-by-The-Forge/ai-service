"""OCR dependency failures keep their real cause (Phase 2 audit F5)."""

import importlib
import sys

import pytest

import app.services.ocr.paddle as paddle_module
from app.core.documents.errors import OCRDependencyUnavailableError
from app.services.ocr.paddle import PaddleOCRService


def test_a_missing_package_is_reported_as_not_installed(monkeypatch):
    cause = ModuleNotFoundError("No module named 'paddleocr'", name="paddleocr")
    monkeypatch.setattr(paddle_module, "PaddleOCR", None)
    monkeypatch.setattr(paddle_module, "_IMPORT_ERROR", cause)

    with pytest.raises(OCRDependencyUnavailableError) as exc:
        PaddleOCRService()

    assert exc.value.missing is True
    assert "paddleocr is not installed (pip install paddleocr)" in str(exc.value)
    assert exc.value.__cause__ is cause
    assert exc.value.error_code == "OCR_DEPENDENCY_UNAVAILABLE"


@pytest.mark.parametrize(
    "cause",
    [
        OSError("[WinError 126] DLL load failed while importing cv2"),
        ModuleNotFoundError("No module named 'shapely'", name="shapely"),
        AttributeError("module 'numpy' has no attribute 'int'"),
    ],
    ids=["native-library", "missing-transitive-dependency", "incompatible-dependency"],
)
def test_an_internal_import_failure_keeps_its_real_cause(monkeypatch, cause):
    monkeypatch.setattr(paddle_module, "PaddleOCR", None)
    monkeypatch.setattr(paddle_module, "_IMPORT_ERROR", cause)

    with pytest.raises(OCRDependencyUnavailableError) as exc:
        PaddleOCRService()

    assert exc.value.missing is False
    assert f"{type(cause).__name__}: {cause}" in str(exc.value)
    assert "not installed" not in str(exc.value)
    assert exc.value.__cause__ is cause


def test_a_model_load_failure_is_typed_and_chained(monkeypatch):
    cause = RuntimeError("inference.pdmodel not found")

    class BrokenPaddleOCR:
        def __init__(self, **kwargs):
            raise cause

    monkeypatch.setattr(paddle_module, "PaddleOCR", BrokenPaddleOCR)

    with pytest.raises(OCRDependencyUnavailableError) as exc:
        PaddleOCRService()

    assert (
        "PaddleOCR model (en, PP-OCRv3) failed to load: RuntimeError: inference.pdmodel not found"
        in str(exc.value)
    )
    assert exc.value.__cause__ is cause


def test_the_import_failure_is_captured_and_logged_at_import(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "paddleocr", None)  # makes `import paddleocr` raise
    try:
        with caplog.at_level("WARNING", logger="app.services.ocr.paddle"):
            reloaded = importlib.reload(paddle_module)
        assert reloaded.PaddleOCR is None
        assert isinstance(reloaded._IMPORT_ERROR, ImportError)
        assert "paddleocr import failed: ModuleNotFoundError" in caplog.text
        assert caplog.records[-1].exc_info is not None
        with pytest.raises(OCRDependencyUnavailableError) as exc:
            reloaded.PaddleOCRService()
        assert exc.value.__cause__ is reloaded._IMPORT_ERROR
    finally:
        monkeypatch.undo()
        importlib.reload(paddle_module)
    assert paddle_module.PaddleOCR is not None or paddle_module._IMPORT_ERROR is not None


def test_a_pymupdf_import_failure_keeps_its_real_cause(monkeypatch):
    import app.services.ocr.pdf_parser as pdf_parser_module

    cause = OSError("[WinError 126] DLL load failed while importing _fitz")
    monkeypatch.setattr(pdf_parser_module, "fitz", None)
    monkeypatch.setattr(pdf_parser_module, "_IMPORT_ERROR", cause)

    with pytest.raises(OCRDependencyUnavailableError) as exc:
        pdf_parser_module.PDFParser().extract_pages("notes.pdf")

    assert exc.value.missing is False
    assert "PyMuPDF failed to load: OSError" in str(exc.value)
    assert exc.value.__cause__ is cause


def test_a_missing_pillow_is_typed_before_the_engine_runs(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "PIL", None)  # makes `from PIL import Image` raise
    image = tmp_path / "page.png"
    image.write_bytes(b"never decoded")

    with pytest.raises(OCRDependencyUnavailableError) as exc:
        paddle_module.validate_image(str(image))

    assert exc.value.missing is True
    assert "Pillow is not installed (pip install Pillow)" in str(exc.value)
    assert isinstance(exc.value.__cause__, ModuleNotFoundError)
