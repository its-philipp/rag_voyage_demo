# ColBERT Overview

ColBERT (Contextualized Late Interaction over BERT) is a retrieval model that balances the expressiveness of cross-encoders with the efficiency of bi-encoders by introducing late interaction. Queries and documents are encoded into token-level embeddings. At search time, ColBERT computes MaxSim between each query token embedding and document token embeddings, then sums these maxima to produce a score. This preserves token-level matching while allowing efficient indexing.
Key properties:
- Late Interaction: preserves rich token semantics during retrieval.
- Efficiency: document embeddings can be precomputed and compressed.
- Accuracy: more precise than single-vector bi-encoders on many IR tasks.
Common use: reranking small candidate sets or building ColBERT-specific ANN indexes.
