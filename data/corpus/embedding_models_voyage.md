# Embedding Models (Voyage)

Voyage models map text to dense vectors. Key considerations:
- Dimensionality: affects index size and compute.
- Throughput/latency: batch requests to improve efficiency.
- Domain adaptation: specialty models may score higher on niche corpora.
For RAG, ensure consistent model usage for index build and querying.
