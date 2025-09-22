import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi


def main():
    """Builds and saves a BM25 index from the project's documents."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_docs.jsonl"
    index_dir = Path(__file__).resolve().parents[1] / "index_bm25"
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
