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
- Baseline comparisons between different RAG approaches

## Quickstart

### Prerequisites
- Python 3.10+
- Voyage AI API key (set as `VOYAGE_API_KEY`)
- Optional: `OPENAI_API_KEY` for evaluation feedback metrics

### Setup
1) Clone + install
```bash
git clone https://github.com/its-philipp/rag_voyage_demo.git
cd rag_voyage_demo
make setup
```
2) Configure
```bash
cp config_example.yaml config.yaml
# edit config.yaml as needed
```
3) Build indices
```bash
make index
.venv/bin/uv run python scripts/build_bm25_index.py
```
4) Query or run API
```bash
make query
# or
make api  # starts Flask on :8000
```
5) Evaluate
```bash
make eval
```

## Docker
Build and run the API:
```bash
make docker-build
make docker-run
# health
curl -s http://localhost:8000/health
# search
curl -s -X POST http://localhost:8000/search -H 'Content-Type: application/json' -d '{"query":"What is ColBERT?"}'
```
CI publishes images to GHCR as `ghcr.io/<owner>/rag-voyage-demo:latest` and `:<git-sha>`.

## Key Scripts
- `build_index.py` — Build FAISS index (reads config; writes to `index/`)
- `query.py` — Hybrid search + reranking pipeline
- `scripts/prepare_data.py` — Generate `data/sample_docs.jsonl` (uses `data/corpus/` if present)
- `scripts/build_bm25_index.py` — Build BM25 index to `index_bm25/`
- `scripts/ingest_folder.py` — Convert a folder of `.md/.txt` to JSONL
- `eval/run_evaluation.py` — Baseline vs Agentic evaluation with metrics

## Config highlights (`config.yaml`)
```yaml
embedding:
  provider: voyage
  model: voyage-2
  dim: 1024
  api_key_env: VOYAGE_API_KEY

faiss:
  type: ivf_pq
  nlist: 16
  m: 32
  nprobe: 16
  nbits: 6

retrieval:
  top_m: 200
  top_k: 20

reranker:
  enabled: true
  type: colbert
  colbert_model: colbert-ir/colbertv2.0
  reranker_k: 10
  device: cpu

index_path: "./index"
bm25_index_path: "./index_bm25"
```

## Data ingestion
- Place `.md`/`.txt` in `data/corpus/` then:
```bash
.venv/bin/uv run python scripts/prepare_data.py
make index
.venv/bin/uv run python scripts/build_bm25_index.py
```
- Or ingest an external folder into JSONL:
```bash
.venv/bin/uv run python scripts/ingest_folder.py /path/to/folder data/sample_docs.jsonl
make index
.venv/bin/uv run python scripts/build_bm25_index.py
```

## Dev commands
```bash
make setup
make index
make query
make api
make eval
make format
make lint
make test
make type
```

## Project layout
```
├── src/                     # Core code
├── scripts/                 # Utilities & ingestion
├── eval/                    # Evaluation & reports
├── data/                    # Sample docs & corpus/
├── index/, index_bm25/      # Built indexes
├── apps/api.py              # Flask API
├── Dockerfile               # Container build
├── .github/workflows/ci.yml # CI: lint/type/test + GHCR
├── config.yaml              # Runtime config
├── pyproject.toml           # Dependencies
└── uv.lock                  # Lockfile
```

## CI/CD
- GitHub Actions: lint (ruff), format check, type (mypy), test (pytest)
- Docker build & publish to GHCR
- Optional: pre-commit for local hooks
```bash
.venv/bin/uv run pre-commit install
.venv/bin/uv run pre-commit run --all-files
```
