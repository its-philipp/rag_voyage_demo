# FAISS IVFPQ

IVFPQ combines an inverted file (IVF) with product quantization (PQ). IVF partitions the space into nlist clusters; queries probe nprobe nearest clusters. PQ compresses residual vectors into subvector codebooks. Trade-offs:
- Pros: memory-efficient, fast search over large corpora.
- Cons: approximate results, requires sufficient training data (often ≥ 4×nlist samples).
Key params: nlist (coarse cells), m (PQ subvectors), nprobe (clusters visited at query time).
