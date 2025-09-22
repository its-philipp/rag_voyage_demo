# ColBERT Latency Considerations

ColBERT incurs per-query encoding and MaxSim computation. To reduce latency: cache frequent queries, cap reranker_k, and ensure efficient batching. On GPU, throughput improves substantially. For small candidate sets, CPU can suffice.
