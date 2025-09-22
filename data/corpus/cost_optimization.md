# Cost Optimization

Cost drivers include embedding volume, reranker inference, and generator tokens. Batch embeddings, cache repeats, limit reranker_k, and use small yet capable LLMs for evaluation. For infra, autoscale, spot/preemptible instances, and consider serverless endpoints for bursty workloads.
