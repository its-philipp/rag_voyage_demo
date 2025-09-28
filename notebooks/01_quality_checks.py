# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install -q pyyaml rank-bm25 faiss-cpu==1.8.0 voyageai
# COMMAND ----------
# Restart Python so newly installed libraries are used in this session
try:
    dbu = globals().get("dbutils")
    if dbu is not None:
        dbu.library.restartPython()
except Exception as e:  # noqa: BLE001
    print("If restart is unavailable, use the blue 'Restart Python' banner or rerun this cell:", e)
# COMMAND ----------
"""
Load VOYAGE_API_KEY from Databricks secret scope if not present.
"""
import os  # noqa: E402

if not os.getenv("VOYAGE_API_KEY"):
    try:
        dbu = globals().get("dbutils")
        if dbu is not None:
            os.environ["VOYAGE_API_KEY"] = dbu.secrets.get(
                scope="rag-voyage-demo", key="VOYAGE_API_KEY"
            )
            print("VOYAGE_API_KEY set from secret scope.")
    except Exception as e:  # noqa: BLE001
        print("Warning: Could not load VOYAGE_API_KEY from secret scope:", e)
# COMMAND ----------
# COMMAND ----------
"""
Quality Checks notebook

Verifies built indexes and runs a couple of sample retrievals (BM25 + FAISS).
"""

from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import json  # noqa: E402
import yaml  # noqa: E402


def _ensure_project_root_on_path() -> Path:
    candidates = []
    try:
        candidates.append(Path(__file__).resolve())
    except NameError:
        pass
    candidates.append(Path.cwd())

    for base in candidates:
        p = base
        for _ in range(6):
            if (
                (p / "pyproject.toml").exists()
                or (p / ".git").exists()
                or ((p / "src").exists() and (p / "apps").exists())
            ):
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))
                return p
            if p.parent == p:
                break
            p = p.parent

    ws_root = Path("/Workspace/Users")
    if ws_root.exists():
        for user_dir in ws_root.iterdir():
            if not user_dir.is_dir():
                continue
            for repo_dir in user_dir.iterdir():
                if not repo_dir.is_dir():
                    continue
                if (repo_dir / "pyproject.toml").exists() and (repo_dir / "src").exists():
                    if str(repo_dir) not in sys.path:
                        sys.path.insert(0, str(repo_dir))
                    return repo_dir

    # Fallback
    p = candidates[0].parent if candidates else Path.cwd()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


PROJECT_ROOT = _ensure_project_root_on_path()
print(f"Project root: {PROJECT_ROOT}")

# COMMAND ----------
# Load config
cfg_path = PROJECT_ROOT / "config.yaml"
cfg = yaml.safe_load(open(cfg_path, "r"))
print("Loaded config from:", cfg_path)

index_dir = Path(cfg.get("index_dir", PROJECT_ROOT / "index"))
if not index_dir.is_absolute():
    index_dir = PROJECT_ROOT / index_dir

bm25_dir = Path(cfg.get("bm25_index_path", PROJECT_ROOT / "index_bm25"))
if not bm25_dir.is_absolute():
    bm25_dir = PROJECT_ROOT / bm25_dir

print("FAISS index dir:", index_dir)
print("BM25 index dir:", bm25_dir)
print("FAISS files:", list(index_dir.glob("*")))
print("BM25 files:", list(bm25_dir.glob("*")))

# COMMAND ----------
from scripts.build_bm25_index import main as build_bm25  # noqa: E402

build_bm25()

# COMMAND ----------
# BM25 sample query using SparseRetriever  # noqa: E402
from src.sparse_retriever import SparseRetriever  # noqa: E402

bm25 = SparseRetriever(str(bm25_dir))
query = "What is ColBERT?"
results = bm25.search(query, top_k=5)
print("BM25 results:")
for doc_id, score in results:
    print(f"{score:.3f}\t{doc_id}")

# COMMAND ----------
# FAISS sample query using Voyage embeddings  # noqa: E402
import os  # noqa: E402
import faiss  # noqa: E402
from src.voyage_client import VoyageClient  # noqa: E402

meta_path = index_dir / "meta.jsonl"
faiss_path = index_dir / "voyage.faiss"
if faiss_path.exists() and meta_path.exists():
    print("Loading FAISS index and metadata...")
    index = faiss.read_index(str(faiss_path))
    meta = [json.loads(line) for line in open(meta_path, "r")]  # order must match vectors

    vc = VoyageClient(
        api_key=os.getenv(cfg["embedding"]["api_key_env"], ""), model=cfg["embedding"]["model"]
    )
    q = "What is ColBERT?"
    q_vec = vc.embed([q]).astype("float32")
    faiss.normalize_L2(q_vec)
    D, top_idx = index.search(q_vec, 5)
    print("FAISS results:")
    for rank, (dist, idx) in enumerate(zip(D[0], top_idx[0]), start=1):
        item = meta[idx]
        print(f"{rank}. {dist:.3f}\t{item.get('doc_id')}\t{item.get('title')}")
else:
    print("FAISS artifacts not found; skipping FAISS sample search.")

# COMMAND ----------
