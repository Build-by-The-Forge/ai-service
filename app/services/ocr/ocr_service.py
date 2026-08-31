from app.services.ocr.cleaner import TextCleaner
from app.services.ocr.paddle import PaddleOCRService
from app.services.ocr.pdf_parser import PDFParser


class OCRService:
    def __init__(self):
        self.ocr = PaddleOCRService()
        self.cleaner = TextCleaner()
        self.pdf_parser = PDFParser()

    async def process_image(self, image_path: str):
        results = await self.ocr.extract_text_from_image(image_path)

        cleaned = [
            {
                **r,
                "text": self.cleaner.clean(r["text"]),
            }
            for r in results
        ]

        return cleaned

    def process_pdf(self, pdf_path: str):
        pages = self.pdf_parser.extract_pages(pdf_path)

        cleaned_pages = [self.cleaner.clean(page) for page in pages]

        return cleaned_pages
