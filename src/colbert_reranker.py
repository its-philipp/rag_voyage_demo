import torch
from typing import List
from colbert.infra import Run
from colbert.modeling.colbert import ColBERT
from colbert.infra.config.config import ColBERTConfig
from colbert.modeling.tokenization.query_tokenization import QueryTokenizer
from colbert.modeling.tokenization.doc_tokenization import DocTokenizer


# Note: For lightweight reranking, we DON'T need to prebuild a ColBERT index if
# we only rerank a small set of candidates. We'll encode on the fly.
# This class wraps scoring of (query, passages) via late interaction.
class ColBERTReranker:
    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        max_query_len: int = 64,
        max_doc_len: int = 180,
        device: str = None,
    ):
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.cfg = ColBERTConfig(
            doc_maxlen=max_doc_len, query_maxlen=max_query_len, nbits=2
        )
        self.model_name = model_name
        self._load()

    def _load(self):
        # initialize Run context and model
        self.run = Run()
        self.colbert = ColBERT(name=self.model_name, colbert_config=self.cfg)
        self.colbert.to(self.device)
        self.colbert.eval()

        # tokenizers
        self.query_tok = QueryTokenizer(self.cfg)
        self.doc_tok = DocTokenizer(self.cfg)

    @torch.no_grad()
    def score(self, query: str, passages: List[str]) -> List[float]:
        # Tokenize
        q_ids, q_mask = self.query_tok.tensorize([query])
        d_ids, d_mask = self.doc_tok.tensorize(passages)

        # Move to device
        q_ids, q_mask = q_ids.to(self.colbert.device), q_mask.to(self.colbert.device)
        d_ids, d_mask = d_ids.to(self.colbert.device), d_mask.to(self.colbert.device)

        # Encode
        Q = self.colbert.query(q_ids, q_mask)
        D = self.colbert.doc(d_ids, d_mask, keep_dims="return_mask")
        if isinstance(D, tuple):
            D, D_mask = D
        else:
            D_mask = d_mask

        scores = self.colbert.score(Q, D, D_mask).cpu().tolist()
        return scores
