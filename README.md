# RAG Voyage Demo: Hybrid Search with Contextual Embeddings + Agentic RAG

A comprehensive RAG (Retrieval-Augmented Generation) system featuring hybrid search, multiple reranking strategies, and agentic capabilities.

## Features

### 🔍 **Hybrid Search Pipeline**
- **Dense Retrieval**: Voyage contextual embeddings with FAISS indexing (IVF-PQ)
- **Sparse Retrieval**: BM25 with rank-bm25
- **Reciprocal Rank Fusion**: Combines dense and sparse results
- **Advanced Reranking**: ColBERT (late interaction) with CrossEncoder fallback

### 🤖 **Agentic RAG**
- Multi-step reasoning with query decomposition
- Iterative retrieval and synthesis
- Context-aware answer generation

### 📊 **Evaluation Framework**
- Comprehensive evaluation metrics (groundedness, relevance, answer quality)
- TruLens integration for automated evaluation
- Baseline comparisons between different RAG approaches

## Quickstart

### Prerequisites
- Python 3.10+
- Voyage AI API key (set as `VOYAGE_API_KEY` environment variable)

### Setup
1. **Clone and setup environment:**
   ```bash
   git clone https://github.com/its-philipp/rag_voyage_demo.git
   cd rag_voyage_demo
   make setup
   ```

2. **Configure:**
   ```bash
   cp config_example.yaml config.yaml
   # Edit config.yaml with your settings
   ```

3. **Build indices:**
   ```bash
   make index
   ```

4. **Test queries:**
   ```bash
   make query
   # Or run specific queries
   echo "What is RAG?" | make query
   ```

5. **Run evaluations:**
   ```bash
   make eval
   ```

## Architecture

### Core Components
- **`src/voyage_client.py`** — Voyage embeddings API client
- **`src/chunking.py`** — Document chunking with context preservation
- **`src/sparse_retriever.py`** — BM25 sparse retrieval
- **`src/colbert_reranker.py`** — ColBERT late-interaction reranking
- **`src/cross_encoder_reranker.py`** — Cross-encoder reranking fallback
- **`src/agentic_rag.py`** — Agentic RAG with multi-step reasoning

### Scripts & Tools
- **`build_index.py`** — Build FAISS and BM25 indices
- **`query.py`** — Query pipeline with hybrid search + reranking
- **`scripts/build_bm25_index.py`** — Build BM25 index
- **`scripts/prepare_data.py`** — Data preparation utilities

### Evaluation
- **`eval/run_evaluation.py`** — Comprehensive evaluation suite
- **`eval/feedback.py`** — Custom evaluation metrics
- **`eval/reports/`** — Baseline evaluation results

## Configuration

Key settings in `config.yaml`:
```yaml
embedding:
  model: voyage-2  # or voyage-context-3
  dim: 1024

retrieval:
  top_m: 200  # Dense/sparse candidates
  top_k: 20   # Final results after reranking

reranker:
  colbert_model: colbert-ir/colbertv2.0
  device: cpu   # or cuda/mps
```

## Performance Notes

- **ColBERT**: Benefits greatly from GPU acceleration
- **Apple Silicon**: May have compatibility issues with PyTorch/ML libraries
- **AMD Systems**: Generally work better for ML workloads
- **CPU Fallback**: All components support CPU execution

## Development

### Available Commands
```bash
make setup     # Install dependencies with uv
make index     # Build FAISS and BM25 indices
make query     # Run sample queries
make eval      # Run evaluation suite
make format    # Format code with black/ruff
make lint      # Run ruff linter
make test      # Run pytest suite
make type      # Run mypy type checking
```

### Project Structure
```
├── src/                    # Core source code
├── scripts/               # Utility scripts
├── eval/                  # Evaluation framework
├── data/                  # Sample documents
├── tests/                 # Test suite
├── config.yaml            # Configuration
├── pyproject.toml         # Dependencies (uv)
└── uv.lock               # Locked dependencies
```
