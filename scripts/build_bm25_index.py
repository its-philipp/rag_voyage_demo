import json
import pickle
from pathlib import Path
import yaml
from rank_bm25 import BM25Okapi


def main():
    """Builds and saves a BM25 index from the project's documents."""
    project_root = Path(__file__).resolve().parents[1]
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
