# RAG Voyage Demo

Hybrid search (dense + sparse) with reranking and agentic RAG. Packaged with CLI, API, Docker, and Databricks/Terraform infra.

## Overview

This is a production-ready RAG (Retrieval-Augmented Generation) system that combines multiple search strategies for optimal document retrieval and answer generation. The system provides:

### 🚀 **Application Interfaces**
- **REST API** (Flask): Two endpoints for health checks and search queries
  - `GET /health` - Service health check
  - `POST /search` - Hybrid search with JSON request/response
- **CLI Tools**: Command-line utilities for building indices and running queries
- **Docker Support**: Containerized deployment with CI/CD via GitHub Actions
- **Databricks Integration**: Terraform-managed infrastructure with automated job scheduling

### 📊 **Evaluation & Metrics**

The system includes a comprehensive evaluation framework that measures RAG quality across three key dimensions:

- **Groundedness** (0-1): Measures whether the generated answer is fully supported by the retrieved source documents (prevents hallucination)
- **Context Relevance** (0-1): Evaluates how relevant the retrieved documents are to the user's question
- **Answer Relevance** (0-1): Assesses how well the final answer addresses the user's question

These metrics are computed using GPT-4o-mini as an evaluator and can compare baseline RAG vs. agentic RAG approaches. Reports are automatically generated in `eval/reports/`.

### 🔄 **Scoring Pipeline**

Search results go through a multi-stage scoring process:
1. **Initial Retrieval**: Dense (FAISS) and Sparse (BM25) retrievers independently fetch candidates
2. **Reciprocal Rank Fusion**: Combines results from both retrievers using rank-based scoring
3. **Reranking**: ColBERT or CrossEncoder models compute final relevance scores
4. **Final Response**: Top-K documents with reranker scores returned to user

## Features

### 🔍 **Hybrid Search Pipeline**
- **Dense Retrieval**: Voyage contextual embeddings with FAISS indexing (IVF-PQ)
- **Sparse Retrieval**: BM25 with rank-bm25
- **Reciprocal Rank Fusion**: Combines dense and sparse results
- **Advanced Reranking**: ColBERT (late interaction) with CrossEncoder fallback

### 🧱 Tech Stack
- Python, uv (packaging), FAISS, rank-bm25
- Voyage AI embeddings
- Transformers / sentence-transformers
- Flask API, Docker
- Terraform + Databricks (repo + jobs + workflow)

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
make query         # runs a sample query via apps/cli/query
make api           # starts Flask on :8000
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

Run with mounts (uses local code and indexes, and your .env):
```bash
docker run --rm -p 8000:8000 --env-file .env \
  -v "$PWD:/app" \
  -v "$PWD/index:/app/index" \
  -v "$PWD/index_bm25:/app/index_bm25" \
  rag-voyage-demo:latest
```
CI publishes images to GHCR as `ghcr.io/<owner>/rag-voyage-demo:latest` and `:<git-sha>`.

## Key Scripts
- `src/index_build.py` — Build FAISS index (reads config; writes to `index/`)
- `src/pipeline.py` — Hybrid search + reranking pipeline
- `scripts/prepare_data.py` — Generate `data/sample_docs.jsonl` (uses `data/corpus/` if present)
- `scripts/build_bm25_index.py` — Build BM25 index to `index_bm25/`
- `scripts/ingest_folder.py` — Convert a folder of `.md/.txt` to JSONL
- `eval/run_evaluation.py` — Baseline vs Agentic evaluation with metrics

## Testing & Report Generation

Generate comprehensive RAG reports showing queries, retrieved chunks, generated answers, and all metrics:

```bash
# Make sure Flask API is running (in one terminal)
make api

# In another terminal, generate report
.venv/bin/uv run python examples/generate_rag_report.py
```

The report generator provides:
- **Retrieved Chunks**: Top documents with ColBERT reranker scores
- **Generated Answers**: LLM responses using retrieved context
- **RAG Triad Metrics**: Groundedness, Context Relevance, Answer Relevance
- **Comprehensive Analysis**: Score distributions, latency, quality assessment
- **JSON Output**: Structured report saved to `examples/rag_report.json`

**Example output:**
```
📊 COMPREHENSIVE SCORE SUMMARY
🔍 RETRIEVAL SCORES (ColBERT Reranker):
  [1]  52.58 - colbert overview
  [2]  50.25 - ColBERT v2
  ...
🎯 RAG TRIAD METRICS (LLM Judge: GPT-4o-mini):
  Groundedness:       🌟 1.000 - Excellent
  Context Relevance:  🌟 1.000 - Excellent
  Answer Relevance:   🌟 1.000 - Excellent
```

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
  enabled: true          # IMPORTANT: Must be true to get real scores
  type: colbert
  colbert_model: colbert-ir/colbertv2.0
  reranker_k: 10
  device: cpu

index_path: "./index"
bm25_index_path: "./index_bm25"
```

**Important:** Set `reranker.enabled: true` in config.yaml to get real ColBERT scores (not placeholder 1.0 values).

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
├── src/                     # Core code (pipeline, index build, rerankers)
├── apps/                    # API & CLI entrypoints
├── scripts/                 # Utilities & ingestion
├── eval/                    # Evaluation & reports
├── examples/                # Demo scripts & generated reports
│   ├── generate_rag_report.py  # Comprehensive RAG report generator
│   └── rag_report.json         # Sample output report
├── data/                    # Sample docs & corpus/
├── index/, index_bm25/      # Built indexes
├── infra/terraform/         # Databricks Terraform
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

## Databricks + Terraform

Provision Databricks resources to run indexing jobs in your workspace.

Prereqs:
- Terraform >= 1.5
- Databricks workspace host and PAT token

Environment:
```bash
export DATABRICKS_HOST="https://adb-<id>.<cloud>.databricks.net"
export DATABRICKS_TOKEN="<pat-token>"
```

Commands:
```bash
make tf-init   # terraform init in infra/terraform
make tf-plan   # shows planned changes
make tf-apply  # creates Databricks repo and jobs
# later
make tf-destroy
```

What it creates:
- Databricks Repo cloned from your git remote/branch
- Two Jobs (standalone) and one Workflow:
  - Build FAISS index (runs apps/cli/build_index.py)
  - Build BM25 index (runs scripts/build_bm25_index.py)
  - Workflow: sequentially runs FAISS → BM25 on the same cluster

Secrets:
- Option A: Pass `-var voyage_api_key=...` in `make tf-plan` / `make tf-apply` to store `VOYAGE_API_KEY` in a Databricks secret scope `rag-voyage-demo`.
- Option B: Create the secret manually via UI/CLI and set the same key.
Jobs read the key from the secret scope when running.

## Databricks Notebooks

Starter notebooks are versioned in `notebooks/` and can be imported into your Databricks Repo:

- `01_quality_checks.py`
  - Installs deps with `%pip`
  - Restarts Python for the session
  - Locates project root in workspace
  - Verifies FAISS (`index/`) and BM25 (`index_bm25/`) artifacts
  - Runs sample BM25 and FAISS queries
  - Tip: If BM25 artifacts are missing, run:
    ```python
    from scripts.build_bm25_index import main as build_bm25
    build_bm25()
    ```

- `02_evaluation.py`
  - Installs deps with `%pip` (pyyaml, voyageai, openai, typing_extensions, python-dotenv)
  - Restarts Python for the session
  - Loads `VOYAGE_API_KEY` / `OPENAI_API_KEY` from secret scope `rag-voyage-demo` when present
  - Demonstrates calling `eval.feedback` metrics (groundedness, relevance)

Notebook compute:
- Attach an all-purpose cluster (recommended: 13.3 LTS ML CPU). Job clusters are ephemeral and won’t appear for notebooks.
- After `%pip` installs, click the blue “Restart Python” prompt or run `dbutils.library.restartPython()`.
