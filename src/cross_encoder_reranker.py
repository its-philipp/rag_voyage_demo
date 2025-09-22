from typing import Any, List
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def score(self, query: str, passages: List[str]) -> List[float]:
        # CrossEncoder expects list[list[str]]; construct pairs accordingly
        pairs: List[List[str]] = [[query, p] for p in passages]
        raw_scores: Any = self.model.predict(pairs)
        # Normalize return type to List[float]
        if hasattr(raw_scores, "tolist"):
            scores_list = raw_scores.tolist()
        else:
            scores_list = raw_scores
        return [float(s) for s in scores_list]
