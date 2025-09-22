import textwrap
from pathlib import Path


BASE_TOPICS = [
    (
        "colbert_overview.md",
        "ColBERT Overview",
        """
        ColBERT (Contextualized Late Interaction over BERT) is a retrieval model that balances the expressiveness of cross-encoders with the efficiency of bi-encoders by introducing late interaction. Queries and documents are encoded into token-level embeddings. At search time, ColBERT computes MaxSim between each query token embedding and document token embeddings, then sums these maxima to produce a score. This preserves token-level matching while allowing efficient indexing.
        Key properties:
        - Late Interaction: preserves rich token semantics during retrieval.
        - Efficiency: document embeddings can be precomputed and compressed.
        - Accuracy: more precise than single-vector bi-encoders on many IR tasks.
        Common use: reranking small candidate sets or building ColBERT-specific ANN indexes.
        """,
    ),
    (
        "colbert_vs_crossencoder.md",
        "ColBERT vs Cross-Encoders",
        """
        Cross-encoders jointly encode the [query, document] pair with full attention, enabling deep interactions but at high computational cost. They excel in reranking but are infeasible for exhaustive retrieval at scale. ColBERT decouples encoding and uses a MaxSim late interaction at query time, yielding faster retrieval while retaining fine-grained matching. Cross-encoders often achieve the highest accuracy per pair, but ColBERT provides a practical accuracy/speed trade-off for retrieval and reranking.
        """,
    ),
    (
        "colbert_vs_biencoder.md",
        "ColBERT vs Bi-Encoders",
        """
        Bi-encoders represent queries and documents as single dense vectors, enabling fast ANN search but losing token-level nuance. ColBERT uses multiple vectors per text and late interaction, improving sensitivity to exact terms and phrasings. This typically increases recall and MRR over bi-encoders on passage retrieval benchmarks while keeping latency manageable.
        """,
    ),
    (
        "faiss_basics.md",
        "FAISS Basics",
        """
        FAISS provides vector indexes for similarity search. IndexFlatL2 and IndexFlatIP perform exact search and serve as accuracy baselines. For large datasets, approximate indexes like IVF, HNSW, and PQ reduce latency and memory. FAISS supports GPU acceleration, training (e.g., k-means for IVF), and quantization techniques for compact storage.
        """,
    ),
    (
        "faiss_ivfpq.md",
        "FAISS IVFPQ",
        """
        IVFPQ combines an inverted file (IVF) with product quantization (PQ). IVF partitions the space into nlist clusters; queries probe nprobe nearest clusters. PQ compresses residual vectors into subvector codebooks. Trade-offs:
        - Pros: memory-efficient, fast search over large corpora.
        - Cons: approximate results, requires sufficient training data (often ≥ 4×nlist samples).
        Key params: nlist (coarse cells), m (PQ subvectors), nprobe (clusters visited at query time).
        """,
    ),
    (
        "faiss_hnsw.md",
        "FAISS HNSW",
        """
        HNSW builds a multi-layer navigable small-world graph. During search, the algorithm traverses from an entry point, refining candidates through neighbors at each layer. Parameters like efSearch and efConstruction control accuracy vs. latency and build time. HNSW performs well for high-recall approximate search with strong empirical performance.
        """,
    ),
    (
        "hybrid_search.md",
        "Hybrid Search",
        """
        Hybrid search fuses dense semantic results with sparse lexical results (e.g., BM25). Reciprocal Rank Fusion (RRF) is a simple, robust technique to combine rank lists. Benefits:
        - Sparse captures exact terms and rare tokens.
        - Dense captures paraphrases and semantics.
        - Fusion improves robustness across query types.
        """,
    ),
    (
        "bm25_basics.md",
        "BM25 Basics",
        """
        BM25 scores documents based on term frequency, inverse document frequency, and document length normalization. It excels at exact keyword matching, making it complementary to dense retrieval. Tokenization quality and stopword handling significantly influence performance.
        """,
    ),
    (
        "rag_triad.md",
        "RAG Triad Evaluation",
        """
        The RAG Triad evaluates: (1) Context Relevance (retrieval quality), (2) Groundedness (answer supported by context), and (3) Answer Relevance (final answer addresses the query). Improvements often come from better retrievers (context relevance), citation/attribution prompting (groundedness), and instruction tuning (answer relevance).
        """,
    ),
    (
        "chunking_strategies.md",
        "Chunking Strategies",
        """
        Chunk granularity influences recall and precision. Smaller chunks improve precision and reduce noise but may fragment context. Overlap preserves coherence. Adding lightweight metadata (titles, headings) as context can guide rerankers. Empirical tuning of sentence windows and overlap often yields gains.
        """,
    ),
    (
        "embedding_models_voyage.md",
        "Embedding Models (Voyage)",
        """
        Voyage models map text to dense vectors. Key considerations:
        - Dimensionality: affects index size and compute.
        - Throughput/latency: batch requests to improve efficiency.
        - Domain adaptation: specialty models may score higher on niche corpora.
        For RAG, ensure consistent model usage for index build and querying.
        """,
    ),
    (
        "rrf_fusion.md",
        "Reciprocal Rank Fusion",
        """
        RRF merges multiple ranked lists by summing 1/(k + rank). It is parameter-light and robust to noise. Typical k ranges 10–60. RRF often improves recall in hybrid search by elevating candidates appearing in either dense or sparse lists.
        """,
    ),
    (
        "reranking_overview.md",
        "Reranking Overview",
        """
        Rerankers refine candidate lists from fast retrievers. Cross-encoders provide strong pairwise scoring but are compute-heavy. ColBERT offers a middle ground with token-level late interaction. Selecting reranker depends on latency budget and quality target.
        """,
    ),
    (
        "prompt_grounding.md",
        "Prompting for Grounding",
        """
        Grounded answers cite or reflect evidence. Techniques:
        - Ask the model to quote, cite, or list supporting snippets.
        - Penalize unsupported claims in system prompts.
        - Use constrained generation with retrieved spans (extract-then-generate).
        These often improve groundedness in the RAG triad.
        """,
    ),
    (
        "retrieval_failures.md",
        "Common Retrieval Failures",
        """
        Failure modes include:
        - Vocabulary mismatch: dense helps; add synonyms, augment data.
        - Ambiguity: query decomposition or disambiguation prompts.
        - Over-chunking: too small windows fragment context.
        - Poor metadata: missing titles harm reranking.
        Diagnostics: manual query audits, coverage tests, and recall at K.
        """,
    ),
    (
        "bm25_tuning.md",
        "BM25 Tuning",
        """
        Tokenization, stopwords, and stemming strongly affect BM25. Consider domain-specific tokenization and custom stopword lists. For code/doc mixtures, splitting on punctuation and camelCase can help. Validate with held-out queries.
        """,
    ),
    (
        "faiss_tuning.md",
        "FAISS Tuning",
        """
        For IVFPQ, ensure sufficient training samples relative to nlist. Increase nprobe for higher recall at the cost of latency. Consider Flat or HNSW for small corpora. Normalize vectors if using inner product. Measure recall@K against a brute-force baseline to validate.
        """,
    ),
    (
        "colbert_latency.md",
        "ColBERT Latency Considerations",
        """
        ColBERT incurs per-query encoding and MaxSim computation. To reduce latency: cache frequent queries, cap reranker_k, and ensure efficient batching. On GPU, throughput improves substantially. For small candidate sets, CPU can suffice.
        """,
    ),
    (
        "hybrid_benefits.md",
        "Benefits of Hybrid",
        """
        Hybrid retrieval improves robustness: sparse catches exact strings (IDs, codes), dense captures paraphrases. RRF is a strong baseline; learned fusion can add modest gains but requires labels. Hybrid also helps when one modality underperforms due to domain shift.
        """,
    ),
    (
        "eval_methodology.md",
        "Evaluation Methodology",
        """
        Use realistic queries and track metrics over time. For RAG triad, keep prompts stable. Compare retrieval-only metrics (recall@K, MRR) separately from end-to-end metrics. Add canary queries to detect regressions in sparse or dense components.
        """,
    ),
    (
        "agentic_decomposition.md",
        "Agentic Query Decomposition",
        """
        Breaking complex questions into sub-queries can improve coverage. However, avoid over-decomposition that dilutes signal. Limit to 5–10 sub-queries, deduplicate contexts, and synthesize with explicit instructions to ground the final answer.
        """,
    ),
    (
        "context_windowing.md",
        "Context Windowing",
        """
        Rerankers benefit from clean passages. Keep chunks within model context limits; for ColBERT, doc_maxlen ~180 is common. For generator LLMs, prune low-signal passages and summarize when necessary to fit within context.
        """,
    ),
    (
        "metadata_design.md",
        "Metadata Design",
        """
        Include titles, headings, and source identifiers. Metadata aids rerankers and UX (citations). Preserve document boundaries and stable doc_ids to avoid evaluation drift.
        """,
    ),
    (
        "negative_sampling.md",
        "Negative Sampling",
        """
        For training or tuning rerankers, hard negatives improve discrimination. Even in evaluation, include queries that test edge cases to avoid overfitting to easy wins.
        """,
    ),
    (
        "index_size_considerations.md",
        "Index Size Considerations",
        """
        Vector dimensionality and number of chunks drive memory. PQ reduces footprint but may reduce accuracy. Track index size and latency alongside quality to find acceptable trade-offs.
        """,
    ),
    (
        "security_privacy.md",
        "Security & Privacy",
        """
        When sending queries to external APIs (embeddings, LLMs), consider PII handling, data minimization, and encryption in transit. Offer opt-out and redaction for sensitive fields.
        """,
    ),
    (
        "cost_optimization.md",
        "Cost Optimization",
        """
        Cost drivers include embedding volume, reranker inference, and generator tokens. Batch embeddings, cache repeats, limit reranker_k, and use small yet capable LLMs for evaluation. For infra, autoscale, spot/preemptible instances, and consider serverless endpoints for bursty workloads.
        """,
    ),
    (
        "databricks_overview.md",
        "Databricks for RAG",
        """
        Databricks provides managed compute, Delta Lake storage, MLflow tracking, and Unity Catalog governance. For RAG: orchestrate ETL to curate corpora, run embedding jobs at scale, store vectors/metadata, and serve retrieval with model-serving. Terraform can provision clusters, storage, and permissions consistently across clouds.
        """,
    ),
    (
        "terraform_basics.md",
        "Terraform Basics",
        """
        Terraform codifies infrastructure. Core concepts: providers, resources, variables, and state. Modules enable reuse. For a RAG stack, define modules for VPC/networking, Databricks workspace, clusters, storage, and secrets. CI plans and applies changes with approvals.
        """,
    ),
    (
        "kubernetes_role.md",
        "Kubernetes Role in RAG",
        """
        Kubernetes is optional. Managed services (Databricks model serving, serverless functions) can suffice. Kubernetes helps if you need custom vector services, GPUs, or multi-tenant workloads. Helm charts and HPA enable repeatability and autoscaling.
        """,
    ),
    (
        "azure_vs_aws.md",
        "Azure vs AWS for RAG",
        """
        Both clouds offer managed vector stores, serverless, and GPUs. Choose based on existing agreements and data residency. On Azure, combine Databricks with Azure OpenAI/Voyage and Blob/Delta. On AWS, use Databricks with Bedrock/Voyage and S3/Delta. Terraform abstracts differences via providers.
        """,
    ),
]


def main():
    root = Path(__file__).resolve().parents[1]
    corpus_dir = root / "data" / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    topics = list(BASE_TOPICS)
    # Expand to ~50 docs by creating focused variants
    variants = [
        ("colbert_practical_tips.md", "ColBERT Practical Tips", "Batch queries, cap doc_maxlen, and cache frequent passages. Monitor MaxSim distributions and layer norms for anomalies.\n"),
        ("faiss_metrics.md", "FAISS Metrics", "Measure recall@K vs. Flat, QPS, and memory footprint. Validate on held-out queries before changing nlist/nprobe."),
        ("bm25_tokenization.md", "BM25 Tokenization", "Customize tokenization for hyphens, underscores, and code identifiers; extend stopword lists for your domain."),
        ("hybrid_rrf_k.md", "RRF k Parameter", "Typical k=60 balances aggressiveness; tune per corpus size. Larger k dampens rank differences."),
        ("eval_canary_queries.md", "Canary Queries", "Include queries targeting rare tokens, synonyms, and numeric identifiers to detect regressions early."),
        ("agentic_limits.md", "Agentic Limits", "Decompose only when necessary to avoid drifting into irrelevant sub-queries; cap at ~10."),
        ("context_window_limits.md", "Context Window Limits", "Stay within generator context limits; summarize or trim low-signal passages when needed."),
        ("reranker_k_tuning.md", "Tuning reranker_k", "Values 10–30 often suffice; increasing beyond may add noise and latency."),
        ("dense_vector_norms.md", "Vector Normalization", "Normalize L2 for inner-product search; ensure consistent preprocessing at index and query time."),
        ("data_quality.md", "Data Quality", "Prefer authoritative, concise docs; remove duplicates; enforce consistent titles and doc_ids."),
        ("index_update_strategies.md", "Index Updates", "Use periodic full rebuilds or append-only with periodic compaction; version artifacts for rollback."),
        ("infra_costs.md", "Infra Costs", "Batch embedding jobs, use spot instances, and autoscale serving layers to control spend."),
        ("security_controls.md", "Security Controls", "Manage secrets via vaults, restrict egress, and audit API usage; redact PII early."),
        ("prompt_citations.md", "Prompting with Citations", "Ask for quotations and cite doc_ids. Enforce that each claim references retrieved evidence."),
        ("bm25_vs_dense.md", "BM25 vs Dense", "BM25 excels at exact tokens and codes; dense handles paraphrase. Hybrid mitigates both weaknesses."),
        ("colbert_gpu.md", "ColBERT on GPU", "Significant latency reductions; ensure CUDA versions match PyTorch and Transformers."),
        ("faiss_ivf_training.md", "IVF Training Data", "Aim for ≥4×nlist training samples; otherwise prefer Flat or HNSW to avoid poor centroids."),
        ("logging_observability.md", "Observability", "Log query latency breakdown (dense, sparse, fusion, reranker) and top-k overlaps."),
    ]
    topics.extend((fn, title, f"""{body}\n""") for fn, title, body in variants)

    for fname, title, body in topics:
        p = corpus_dir / fname
        content = f"# {title}\n\n" + textwrap.dedent(body).strip() + "\n"
        p.write_text(content, encoding="utf-8")

    print(f"Seeded {len(topics)} docs into {corpus_dir}")


if __name__ == "__main__":
    main()


