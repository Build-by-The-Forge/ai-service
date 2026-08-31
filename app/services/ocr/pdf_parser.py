try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - dependency optional in some environments
    fitz = None


class PDFParser:
    def extract_pages(self, pdf_path: str) -> list[str]:
        if fitz is None:
            raise RuntimeError("PyMuPDF is required to extract PDF text")

        doc = fitz.open(pdf_path)

        pages: list[str] = []
        for page in doc:
            text = page.get_text("text")
            pages.append(text)

        return pages
