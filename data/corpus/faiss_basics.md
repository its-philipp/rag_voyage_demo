# FAISS Basics

FAISS provides vector indexes for similarity search. IndexFlatL2 and IndexFlatIP perform exact search and serve as accuracy baselines. For large datasets, approximate indexes like IVF, HNSW, and PQ reduce latency and memory. FAISS supports GPU acceleration, training (e.g., k-means for IVF), and quantization techniques for compact storage.
