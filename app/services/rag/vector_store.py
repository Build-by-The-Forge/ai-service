class VectorStore:
    def add(self, embedding: list[float], metadata: dict) -> None:
        pass

    def query(self, vector: list[float], top_k: int = 5) -> list[dict]:
        return []
