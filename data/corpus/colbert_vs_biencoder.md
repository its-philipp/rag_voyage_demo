# ColBERT vs Bi-Encoders

Bi-encoders represent queries and documents as single dense vectors, enabling fast ANN search but losing token-level nuance. ColBERT uses multiple vectors per text and late interaction, improving sensitivity to exact terms and phrasings. This typically increases recall and MRR over bi-encoders on passage retrieval benchmarks while keeping latency manageable.
