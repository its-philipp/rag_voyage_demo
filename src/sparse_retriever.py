import pickle
from pathlib import Path
from typing import List, Tuple


class SparseRetriever:
    def __init__(self, index_dir: str):
        """
        Initializes the SparseRetriever by loading a saved BM25 index.

        Args:
            index_dir: Path to the directory containing the BM25 index files.
        """
        self.index_dir = Path(index_dir)
        self._load_index()

    def _load_index(self):
        """Loads the BM25 index and doc IDs from pickle files."""
        index_path = self.index_dir / "bm25_index.pkl"
        with open(index_path, "rb") as f:
            self.bm25 = pickle.load(f)

        ids_path = self.index_dir / "bm25_doc_ids.pkl"
        with open(ids_path, "rb") as f:
            self.doc_ids = pickle.load(f)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Performs a BM25 search for a given query.

        Args:
            query: The search query string.
            top_k: The number of top results to return.

        Returns:
            A list of tuples, where each tuple contains a doc_id and its BM25 score.
        """
        # Simple whitespace tokenizer for the query
        tokenized_query = query.split(" ")

        # Get scores for all documents
        doc_scores = self.bm25.get_scores(tokenized_query)

        # Get the top k results
        top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[
            :top_k
        ]

        results = []
        for i in top_n_indices:
            # Only include results with a score > 0
            if doc_scores[i] > 0:
                results.append((self.doc_ids[i], doc_scores[i]))

        return results
