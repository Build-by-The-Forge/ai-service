from typing import List
import fitz  # PyMuPDF


class PDFParser:

    def extract_pages(self, pdf_path: str) -> List[str]:
        doc = fitz.open(pdf_path)

        pages: List[str] = []
        for page in doc:
            text = page.get_text("text")
            pages.append(text)

        return pages
