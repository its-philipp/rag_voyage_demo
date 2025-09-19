# RAG Demo: Contextual Embeddings (Voyage) + FAISS + ColBERT Reranker

## Quickstart
1) Create a `config.yaml` from `config_example.yaml` and set your `VOYAGE_API_KEY` in the environment.
2) Install deps (ideally in a venv with Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```
3) Build the ANN index:
   ```bash
   python build_index.py
   ```
4) Run a sample query:
   ```bash
   python query.py
   ```

This pipeline:
- Chunks documents and builds **contextual chunk embeddings** using `voyage-context-3`.
- Indexes those vectors in **FAISS** (default IVF-PQ).
- Retrieves top-M candidates and **reranks with ColBERT** (late interaction). A cross-encoder fallback is included.

> NOTE: ColBERT reranking benefits from a GPU but will run on CPU with reduced speed.

## Files
- `src/voyage_client.py` — minimal client for Voyage embeddings
- `src/chunking.py` — simple sentence-based chunking with lightweight title context
- `build_index.py` — embeds chunks and builds FAISS
- `src/colbert_reranker.py` — on-the-fly ColBERT late-interaction reranker
- `src/cross_encoder_reranker.py` — optional cross-encoder reranker
- `query.py` — query-time retrieval + rerank demo
