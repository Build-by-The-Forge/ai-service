import re


class TextCleaner:

    def clean(self, text: str) -> str:
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\.,;:()\-]", "", text)

        return text.strip()
