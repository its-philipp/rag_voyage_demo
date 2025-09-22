import json
from pathlib import Path
import glob

# --- Source Data ---

# Source 1: Pinecone's excellent ColBERT v2 guide
# https://www.pinecone.io/learn/colbert/
colbert_text = """
ColBERTv2 is a retrieval model that balances the expressiveness of computationally expensive models with the efficiency of less expressive but faster models. It introduces a "late interaction" mechanism over BERT-based embeddings.

Standard dense retrieval models, often called bi-encoders, independently embed the query and the document into single-vector representations. During search, they compute a single similarity score (like dot-product or cosine similarity) between the query vector and all document vectors. This is fast but can lose nuance, as the embedding must compress the entire meaning of the text into one vector.

On the other hand, cross-encoders (or re-rankers) jointly process the query and a document. This allows for deeper, token-level comparisons, resulting in much higher accuracy. However, this process is extremely slow and not feasible for searching over millions of documents.

ColBERT offers a middle ground. It produces multi-vector embeddings for both the query and the document. For a query, each of its tokens is converted into an embedding. For a document, ColBERT also embeds each of its tokens. The "late interaction" happens at search time. For each query embedding, ColBERT finds the maximum similarity score against all document embeddings. These maximum scores are then summed up to get the final query-document score.

This approach is more granular than bi-encoders because it allows for token-level matching, but it's much faster than cross-encoders because the expensive joint processing is replaced by a simple MaxSim operation. ColBERTv2 further optimizes this by pre-computing document embeddings and using a compressed representation for the index, making it both fast and memory-efficient.
""".strip()

# Source 2: "The Missing Manual" for FAISS
# https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks
faiss_text = """
FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.

The most basic FAISS index is the `IndexFlatL2`. It performs an exhaustive search, computing the L2 distance between the query vector and every other vector in the index. This is a brute-force approach and serves as the gold standard for accuracy. However, its search time scales linearly with the number of documents, making it slow for large datasets.

For larger datasets, FAISS offers methods that trade some accuracy for speed. A popular choice is the `IndexIVFPQ`. This index works by partitioning the vector space into cells (voronoi cells).
- IVF (Inverted File): At search time, the index only searches within a few cells closest to the query vector, rather than the entire dataset. The `nprobe` parameter controls how many cells to visit. A higher `nprobe` means better accuracy but slower search.
- PQ (Product Quantization): This technique compresses the vectors to reduce their memory footprint. It breaks down a large vector into smaller sub-vectors and quantizes each sub-vector separately. This allows FAISS to store massive datasets in RAM.

In summary, `IndexFlatL2` guarantees finding the exact nearest neighbors but is slow. `IndexIVFPQ` is much faster and more memory-efficient because it uses an approximate search strategy with partitioning and compression, but the results are not guaranteed to be perfect. The trade-off between speed and accuracy is controlled by parameters like `nprobe` and the PQ settings.
""".strip()

# Source 3: Key ideas from the original RAG paper by Meta/Facebook
# https://arxiv.org/abs/2005.11441
rag_text = """
Retrieval-Augmented Generation (RAG) is a methodology for improving the performance of Large Language Models (LLMs) on knowledge-intensive tasks. It combines a pre-trained parametric memory (the LLM itself) with a non-parametric memory, which is typically a dense vector index of a large corpus like Wikipedia.

The core idea is to retrieve relevant documents from the external knowledge source and provide them as context to the LLM when generating an answer. This has several advantages. First, it allows the model to access up-to-date, real-time information that was not available in its training data. Second, it provides a way to ground the model's generations in verifiable evidence, reducing the risk of hallucination and making the outputs more trustworthy.

The RAG framework consists of two main components: a retriever and a generator.
- The Retriever: Given a user input (e.g., a question), the retriever's job is to find a small set of relevant text passages from the knowledge base. This is often implemented using a dense passage retriever (DPR) model, which embeds both the input and the passages into a shared high-dimensional vector space for similarity search.
- The Generator: This is a sequence-to-sequence model (like BART or T5) that takes the original user input and the retrieved passages as context, and generates the final text output. The generator learns to synthesize an answer by conditioning on the provided knowledge.

RAG models can be trained end-to-end, allowing both the retriever and the generator to be fine-tuned together for the specific task. This joint training helps the retriever learn to find passages that are most useful for the generator.
""".strip()

# Source 4: Pinecone on Dense Retrieval
# https://www.pinecone.io/learn/dense-vector-embeddings-for-rag/
dense_retrieval_text = """
Dense retrieval is a method used in information retrieval that represents documents and queries as dense vectors, known as embeddings. Unlike sparse retrieval methods like TF-IDF or BM25, which rely on keyword matching, dense retrieval captures the semantic meaning of the text.

The process begins with an embedding model, often a pre-trained transformer like BERT or a specialized model like Voyage, which maps text to a high-dimensional vector space. In this space, texts with similar meanings are located closer to each other. When a query comes in, it is also embedded into the same vector space. The retrieval system then searches for the document vectors that are closest to the query vector, typically using a similarity measure like cosine similarity or dot product.

The key advantage of dense retrieval is its ability to handle synonyms and semantically related terms. For example, a search for "how to make a car go faster" could retrieve a document about "automobile performance tuning," even if they don't share keywords. This is because the embedding model understands the underlying concepts.

However, dense retrieval can sometimes struggle with queries that require exact keyword matching, such as searching for a specific error code or a person's name. This is where hybrid approaches, which combine dense and sparse retrieval, can be particularly effective.
""".strip()

# Source 5: TruLens on RAG Evaluation
# https://www.trulens.org/trulens_eval/getting_started/core_concepts/rag_triad/
rag_eval_text = """
Evaluating a Retrieval-Augmented Generation (RAG) application involves assessing the quality of both the retrieval and generation components. A useful framework for this is the RAG Triad, which focuses on three key metrics: Answer Relevance, Context Relevance, and Groundedness.

1.  **Context Relevance**: This metric measures how relevant the retrieved context is to the user's query. It answers the question: "Does the retrieved information have a high signal-to-noise ratio for answering the query?" If the context is irrelevant, the generator will struggle to produce a good answer, even if the generator itself is powerful. This is often the first and most important bottleneck to address.

2.  **Groundedness**: This metric assesses how well the generated answer is supported by the retrieved context. It helps detect "hallucinations," where the model makes up information that is not present in the provided text. A high groundedness score means the answer is based on the evidence provided, making it more factual and trustworthy.

3.  **Answer Relevance**: This measures how relevant the final generated answer is to the user's original query. It ensures that the model is not just producing a factually correct statement but is actually addressing the specific question that was asked.

By evaluating these three components separately, developers can pinpoint weaknesses in their RAG system. For example, low context relevance points to a problem with the retriever, while low groundedness with high context relevance points to a problem with the generator's synthesis process.
""".strip()

# Source 6: Elastic on Hybrid Search
# https://www.elastic.co/what-is/hybrid-search
hybrid_search_text = """
Hybrid search is an approach that combines the strengths of traditional keyword-based (sparse) search with modern semantic (dense) vector search. This combination provides more relevant and accurate results than either method could achieve on its own.

Sparse retrieval, powered by algorithms like BM25, excels at finding documents with exact keyword matches. It's highly effective for queries containing specific terms, acronyms, or identifiers. However, it can fail when queries use different wording or synonyms for the concepts present in the documents.

Dense retrieval, on the other hand, uses machine learning models to create vector embeddings that capture the semantic meaning of text. This allows it to find conceptually related documents even if they don't share the same keywords. Its weakness, however, is that it can sometimes miss documents with important keyword matches if the semantic meaning is not perfectly aligned.

Hybrid search resolves this by running both types of queries simultaneously. The results from each are then combined using a fusion algorithm, such as Reciprocal Rank Fusion (RRF). RRF is a technique that ranks the combined results in a way that balances both the keyword and semantic relevance. The result is a search system that is both precise with keywords and intelligent about meaning, delivering a superior user experience.
""".strip()

# --- Script Logic ---


def load_corpus_from_folder(corpus_dir: Path):
    """Load .md/.txt files from corpus_dir into doc dicts with incremental IDs."""
    docs = []
    patterns = ["*.md", "*.markdown", "*.txt"]
    files = []
    for pat in patterns:
        files.extend(sorted(corpus_dir.rglob(pat)))
    for i, fp in enumerate(files, start=1):
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        title = fp.stem.replace("_", " ").replace("-", " ")
        doc = {"doc_id": f"corpus_{i:06d}", "title": title, "text": text}
        docs.append(doc)
    return docs


def main():
    """Create a new sample_docs.jsonl or expand it with data/corpus/* files if present."""
    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "data" / "sample_docs.jsonl"
    out_path.parent.mkdir(exist_ok=True)

    # Start with built-in high-quality docs
    docs = [
        {"doc_id": "doc_000001", "title": "ColBERT v2", "text": colbert_text},
        {"doc_id": "doc_000002", "title": "FAISS Indexes", "text": faiss_text},
        {"doc_id": "doc_000003", "title": "Retrieval-Augmented Generation (RAG)", "text": rag_text},
        {"doc_id": "doc_000004", "title": "Dense Retrieval", "text": dense_retrieval_text},
        {"doc_id": "doc_000005", "title": "RAG Evaluation", "text": rag_eval_text},
        {"doc_id": "doc_000006", "title": "Hybrid Search", "text": hybrid_search_text},
    ]

    # If user has added more high-quality docs under data/corpus/, append them
    corpus_dir = project_root / "data" / "corpus"
    if corpus_dir.exists():
        extra = load_corpus_from_folder(corpus_dir)
        # Avoid doc_id collisions by continuing the sequence
        next_id = len(docs) + 1
        for idx, doc in enumerate(extra, start=next_id):
            doc["doc_id"] = f"doc_{idx:06d}"
        docs.extend(extra)

    with out_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Wrote {len(docs)} documents to {out_path}")


if __name__ == "__main__":
    main()
