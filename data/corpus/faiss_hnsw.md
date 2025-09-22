# FAISS HNSW

HNSW builds a multi-layer navigable small-world graph. During search, the algorithm traverses from an entry point, refining candidates through neighbors at each layer. Parameters like efSearch and efConstruction control accuracy vs. latency and build time. HNSW performs well for high-recall approximate search with strong empirical performance.
