import pytest

from app.services.ocr.ocr_service import OCRService


class DummyOCR:
    async def extract_text_from_image(self, image_path: str):
        return [
            {
                "text": "Hello\nWorld",
                "confidence": 0.99,
                "bbox": [[0, 0], [10, 0], [10, 10], [0, 10]],
            }
        ]


class DummyCleaner:
    def clean(self, text: str) -> str:
        return text.replace("\n", " ").upper()


class DummyParser:
    def extract_pages(self, pdf_path: str):
        return ["page one", "page two"]


@pytest.mark.asyncio
async def test_process_image_cleans_extracted_text(monkeypatch):
    service = OCRService()
    monkeypatch.setattr(service, "ocr", DummyOCR())
    monkeypatch.setattr(service, "cleaner", DummyCleaner())

    result = await service.process_image("image.png")

    assert result == [
        {
            "text": "HELLO WORLD",
            "confidence": 0.99,
            "bbox": [[0, 0], [10, 0], [10, 10], [0, 10]],
        }
    ]


def test_process_pdf_cleans_extracted_pages(monkeypatch):
    service = OCRService()
    monkeypatch.setattr(service, "pdf_parser", DummyParser())
    monkeypatch.setattr(service, "cleaner", DummyCleaner())

    result = service.process_pdf("document.pdf")

    assert result == ["PAGE ONE", "PAGE TWO"]
