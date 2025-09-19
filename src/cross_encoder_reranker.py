from typing import List
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def score(self, query: str, passages: List[str]) -> List[float]:
        pairs = [(query, p) for p in passages]
        scores = self.model.predict(pairs).tolist()
        return scores
