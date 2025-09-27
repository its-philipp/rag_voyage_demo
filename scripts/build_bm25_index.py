import json
import pickle
from pathlib import Path
import yaml
from rank_bm25 import BM25Okapi


def _detect_project_root() -> Path:
    """Detect project root robustly for Databricks/ipykernel contexts.

    Tries multiple bases (path of current file, CWD) and walks up looking for
    markers like pyproject.toml, .git, or the presence of both src/ and apps/.
    """
    candidates = []
    try:
        candidates.append(Path(__file__).resolve())
    except NameError:
        # __file__ may be undefined in some Databricks execution paths
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
                return p
            if p.parent == p:
                break
            p = p.parent

    # Fallback: assume parent of base is the repository
    return candidates[0].parent if candidates else Path.cwd()


def main():
    """Builds and saves a BM25 index from the project's documents."""
    project_root = _detect_project_root()
    cfg = yaml.safe_load(open(project_root / "config.yaml", "r"))
    # Read from config if relative path; otherwise use absolute
    data_path = Path(cfg.get("data_path", project_root / "data" / "sample_docs.jsonl"))
    if not data_path.is_absolute():
        data_path = project_root / data_path
    index_dir = Path(cfg.get("bm25_index_path", project_root / "index_bm25"))
    if not index_dir.is_absolute():
        index_dir = project_root / index_dir
    index_dir.mkdir(exist_ok=True)

    print(f"Loading documents from {data_path}...")
    docs = [json.loads(line) for line in open(data_path)]
    corpus = [doc["text"] for doc in docs]
    doc_ids = [doc["doc_id"] for doc in docs]

    # Simple whitespace tokenizer
    tokenized_corpus = [doc.split(" ") for doc in corpus]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    # Save the index and the doc_ids mapping
    index_path = index_dir / "bm25_index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump(bm25, f)

    ids_path = index_dir / "bm25_doc_ids.pkl"
    with open(ids_path, "wb") as f:
        pickle.dump(doc_ids, f)

    print(f"BM25 index with {len(doc_ids)} documents saved to {index_dir}")


if __name__ == "__main__":
    main()
