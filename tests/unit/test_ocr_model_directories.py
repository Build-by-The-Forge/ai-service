"""PaddleOCR uses the verified, installed models; missing ones fail explicitly
(Phase 2 Step 2)."""

import json

import pytest

import app.services.ocr.paddle as paddle_module
from app.core.documents.errors import OCRDependencyUnavailableError
from app.services.ocr.paddle import MODEL_FILES, PaddleOCRService, model_directories

NAMES = {"det": "en_det", "rec": "en_rec", "cls": "cls_m"}


@pytest.fixture
def installed(tmp_path):
    for role, name in NAMES.items():
        directory = tmp_path / role / name
        directory.mkdir(parents=True)
        for file in MODEL_FILES:
            (directory / file).write_bytes(b"x")
    (tmp_path / "installed.json").write_text(
        json.dumps(
            {"models": {role: {"name": name, "sha256": "0" * 64} for role, name in NAMES.items()}}
        )
    )
    return tmp_path


class RecordingEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_the_installed_directories_are_passed_to_paddleocr(monkeypatch, installed):
    monkeypatch.setattr(paddle_module.settings, "OCR_MODEL_DIR", str(installed))
    monkeypatch.setattr(paddle_module, "PaddleOCR", RecordingEngine)

    kwargs = PaddleOCRService().model.kwargs

    assert kwargs["det_model_dir"] == str(installed / "det" / "en_det")
    assert kwargs["rec_model_dir"] == str(installed / "rec" / "en_rec")
    assert kwargs["cls_model_dir"] == str(installed / "cls" / "cls_m")
    assert (kwargs["lang"], kwargs["ocr_version"]) == ("en", "PP-OCRv3")


@pytest.mark.parametrize("role", ["det", "rec", "cls"])
@pytest.mark.parametrize("file", MODEL_FILES)
def test_a_missing_model_file_is_named(installed, role, file):
    (installed / role / NAMES[role] / file).unlink()
    with pytest.raises(OCRDependencyUnavailableError) as exc:
        model_directories(str(installed))
    assert file in str(exc.value) and NAMES[role] in str(exc.value)
    assert isinstance(exc.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize(
    "record", [None, "not json", json.dumps({"models": {"det": {"name": "x"}}})]
)
def test_an_unreadable_install_record_is_refused(installed, record):
    path = installed / "installed.json"
    if record is None:
        path.unlink()
    else:
        path.write_text(record)
    with pytest.raises(OCRDependencyUnavailableError, match="installed.json unreadable"):
        model_directories(str(installed))


def test_an_empty_model_directory_never_reaches_the_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(paddle_module.settings, "OCR_MODEL_DIR", str(tmp_path))
    constructed = []
    monkeypatch.setattr(paddle_module, "PaddleOCR", lambda **kw: constructed.append(kw))
    with pytest.raises(OCRDependencyUnavailableError):
        PaddleOCRService()
    assert constructed == []


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_strict_environments_require_ocr_model_dir(monkeypatch, environment):
    monkeypatch.setattr(paddle_module.settings, "APP_ENVIRONMENT", environment)
    with pytest.raises(OCRDependencyUnavailableError, match="OCR_MODEL_DIR is required"):
        model_directories(None)


def test_development_without_ocr_model_dir_uses_the_cache_loudly(monkeypatch, caplog):
    monkeypatch.setattr(paddle_module.settings, "APP_ENVIRONMENT", "development")
    with caplog.at_level("WARNING", logger="app.services.ocr.paddle"):
        assert model_directories(None) == {}
    assert "OCR_MODEL_DIR is not set" in caplog.text
