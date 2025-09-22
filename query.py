import os
import json
import yaml
import numpy as np
import faiss
import logging
import time
from typing import Dict, Any, Tuple, List
from src.voyage_client import VoyageClient
from src.cross_encoder_reranker import CrossEncoderReranker
from src.colbert_reranker import ColBERTReranker
from src.sparse_retriever import SparseRetriever
import argparse
from dotenv import load_dotenv


def load_index(cfg: Dict[str, Any]) -> Tuple[faiss.Index, List[Dict]]:
    logger = logging.getLogger(__name__)
    index_path = os.path.join(cfg["index_dir"], "voyage.faiss")
    meta_path = os.path.join(cfg["index_dir"], "meta.jsonl")
    index = faiss.read_index(index_path)
    with open(meta_path, "r") as fh:
        meta = [json.loads(line) for line in fh]
    logger.info(f"Index loaded: {cfg['index_dir']} num_chks={len(meta)}")
    return index, meta


def reciprocal_rank_fusion(
    results_list: List[List[Tuple[str, float]]], k: int = 60
) -> Dict[str, float]:
    fused_scores: Dict[str, float] = {}
    for results in results_list:
        for i, (doc_id, _) in enumerate(results):
            rank = i + 1
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
            fused_scores[doc_id] += 1.0 / (k + rank)

    return dict(sorted(fused_scores.items(), key=lambda item: item[1], reverse=True))


def query_system(query: str, cfg: Dict[str, Any]) -> List[Tuple[float, Dict]]:
    """
    Main query pipeline: Hybrid Search (dense + sparse) -> Reranking
    """
    embedding_model = cfg.get("embedding", {}).get("model")
    retrieval_top_m = int(cfg.get("retrieval", {}).get("top_m", 50))

    # 1. Dense Retrieval (FAISS)
    start_time = time.time()
    query_embedding = voyage_client.get_embedding(query, model=embedding_model)
    D, indices = index.search(np.array([query_embedding]), k=retrieval_top_m)
    logger.info(f"ANN search took: {(time.time() - start_time) * 1000:.2f} ms")

    dense_results: List[Tuple[str, float]] = []
    for i in indices[0]:
        if i != -1:
            dense_results.append((meta[i]["doc_id"], 1.0))

    # 2. Sparse Retrieval (BM25)
    sparse_retriever = SparseRetriever(index_dir=cfg["bm25_index_path"])
    sparse_results = sparse_retriever.search(query, top_k=retrieval_top_m)

    # 3. Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion([dense_results, sparse_results])
    fused_doc_ids = list(fused_results.keys())

    all_docs: Dict[str, Dict] = {}
    for item in meta:
        if item["doc_id"] not in all_docs:
            all_docs[item["doc_id"]] = {
                "doc_id": item["doc_id"],
                "text": item["text"],
                "title": item["title"],
            }

    docs_to_rerank: List[Tuple[str, str]] = []
    for doc_id in fused_doc_ids:
        if doc_id in all_docs:
            docs_to_rerank.append((all_docs[doc_id]["doc_id"], all_docs[doc_id]["text"]))

    # 4. Reranking (support both ColBERT and CrossEncoder interfaces)
    reranker_cfg = cfg.get("reranker")
    if reranker_cfg and reranker_cfg.get("enabled", False):
        reranker_model = reranker_cfg.get("reranker_model")
        reranker_k = reranker_cfg.get("reranker_k", 10)
        device = reranker_cfg.get("device")

        reranker = CrossEncoderReranker(reranker_model, device=device)

        logger.info(f"Reranking with {reranker_model}...")
        reranked_chunks = reranker(query, docs_to_rerank)[:reranker_k]
        final_chunks = reranked_chunks
    else:
        final_chunks = docs_to_rerank

    return final_chunks


# --- Global Inits ---
cfg = yaml.safe_load(open("config.yaml"))
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

index, meta = load_index(cfg)
voyage_client = VoyageClient(os.getenv("VOYAGE_API_KEY"))

# Choose reranker per env or fallback
use_cross = os.getenv("RERANKER", "").lower() == "crossencoder"
reranker: Any
if not use_cross:
    try:
        colbert_model_name = cfg.get("reranker", {}).get("colbert_model") or os.getenv(
            "COLBERT_MODEL", "colbert-ir/colbertv2.0"
        )
        reranker = ColBERTReranker(model_name=colbert_model_name, device=cfg.get("device", "cpu"))
    except Exception as e:
        logging.warning(f"ColBERT init failed ({e}); falling back to CrossEncoder.")
        reranker = CrossEncoderReranker()
else:
    reranker = CrossEncoderReranker()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="The query to search for.")
    args = parser.parse_args()

    results = query_system(args.query, cfg)
    for s, c in results:
        print(f"{s:.3f}\t{c['doc_id']}\t{c['text'][:120]}...")
