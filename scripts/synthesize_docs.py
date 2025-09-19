"""Generate synthetic sample_docs.jsonl for testing index building.
Produces documents with multiple sentences per doc. Script is deterministic by default for reproducibility.

Usage:
  python scripts/synthesize_docs.py --num 1000 --min-sents 3 --max-sents 6
"""
import argparse
import json
import random
from pathlib import Path
from textwrap import shorten

OUT = Path(__file__).resolve().parents[1] / "data" / "sample_docs.jsonl"
OUT.parent.mkdir(exist_ok=True)

SENTENCE_TEMPLATES = [
    "{topic} is an important concept in information retrieval.",
    "It covers core ideas such as {idea1} and {idea2}.",
    "Researchers often evaluate {topic} with retrieval metrics and benchmarks.",
    "Practical systems use {topic} alongside techniques like {idea1} for improved performance.",
    "Recent advances include better {idea1} and more efficient {idea2} approaches.",
    "Applications of {topic} include search, question answering, and knowledge retrieval.",
    "Combining {topic} with reranking improves final answer quality.",
]

TOPICS = [
    "colbert", "faiss", "dense retrievers", "contextual chunking", "RAG", "embeddings", "indexing", "approximate nearest neighbors",
    "semantic search", "vector databases"
]

IDEAS = [
    "late interaction", "embedding normalization", "IVF-PQ", "HNSW", "vector quantization",
    "sentence windowing", "reranking", "sharding", "ann search"
]


def make_doc(i, min_sents, max_sents):
    topic = random.choice(TOPICS)
    idea1, idea2 = random.sample(IDEAS, 2)
    n = random.randint(min_sents, max_sents)
    sents = []
    # choose n templates and format them
    for _ in range(n):
        tmpl = random.choice(SENTENCE_TEMPLATES)
        sents.append(tmpl.format(topic=topic, idea1=idea1, idea2=idea2))
    text = " ".join(sents)
    # shorten extremely long texts for safety
    text = shorten(text, width=800, placeholder="...")
    return {
        "doc_id": f"doc_{i:06d}",
        "title": topic.title(),
        "text": text
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1000, help="number of documents to generate")
    parser.add_argument("--min-sents", type=int, default=3, help="minimum sentences per doc")
    parser.add_argument("--max-sents", type=int, default=6, help="maximum sentences per doc")
    parser.add_argument("--seed", type=int, default=0, help="random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    with OUT.open("w") as f:
        for i in range(1, args.num + 1):
            doc = make_doc(i, args.min_sents, args.max_sents)
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Wrote {OUT} with {args.num} docs (each {args.min_sents}-{args.max_sents} sentences)")


if __name__ == "__main__":
    main()
