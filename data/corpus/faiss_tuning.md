# FAISS Tuning

For IVFPQ, ensure sufficient training samples relative to nlist. Increase nprobe for higher recall at the cost of latency. Consider Flat or HNSW for small corpora. Normalize vectors if using inner product. Measure recall@K against a brute-force baseline to validate.
