# moved from root query.py
import os
import json
import yaml
import numpy as np
import faiss
import logging
import time
from typing import Dict, Any, Tuple, List
from dotenv import load_dotenv

from src.voyage_client import VoyageClient
from src.cross_encoder_reranker import CrossEncoderReranker
from src.colbert_reranker import ColBERTReranker
from src.sparse_retriever import SparseRetriever


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
        for i, (doc_id, _score) in enumerate(results):
            rank = i + 1
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return dict(sorted(fused_scores.items(), key=lambda item: item[1], reverse=True))


def query_system(query: str, cfg: Dict[str, Any]) -> List[Tuple[float, Dict]]:
    retrieval_top_m = int(cfg.get("retrieval", {}).get("top_m", 50))

    start_time = time.time()
    query_embedding = voyage_client.embed([query])[0]
    D, indices = index.search(np.array([query_embedding]), k=retrieval_top_m)
    logger.info(f"ANN search took: {(time.time() - start_time) * 1000:.2f} ms")

    dense_results: List[Tuple[str, float]] = []
    for i in indices[0]:
        if i != -1:
            dense_results.append((meta[i]["doc_id"], 1.0))

    sparse_retriever = SparseRetriever(index_dir=cfg["bm25_index_path"])
    sparse_results = sparse_retriever.search(query, top_k=retrieval_top_m)

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

    reranker_cfg = cfg.get("reranker", {}) or {}
    reranker_enabled = bool(reranker_cfg.get("enabled", False))
    env_choice = os.getenv("RERANKER", "").strip().lower()
    cfg_choice = reranker_cfg.get("type")
    choice = env_choice or (cfg_choice or ("crossencoder" if reranker_enabled else "none"))

    if choice == "colbert" and reranker_enabled:
        reranker_k = int(reranker_cfg.get("reranker_k", 10))
        colbert_model = reranker_cfg.get("colbert_model", "colbert-ir/colbertv2.0")
        device = reranker_cfg.get("device") or cfg.get("device") or None
        passages = [text for (_doc_id, text) in docs_to_rerank]
        if passages:
            cb_reranker = ColBERTReranker(model_name=colbert_model, device=device)
            scores = cb_reranker.score(query, passages)
            scored = list(zip(scores, [all_docs[doc_id] for (doc_id, _text) in docs_to_rerank]))
            reranked_chunks: List[Tuple[float, Dict]] = sorted(
                scored, key=lambda x: x[0], reverse=True
            )[:reranker_k]
            final_chunks = reranked_chunks
        else:
            final_chunks = []
    elif choice == "crossencoder" and reranker_enabled:
        reranker_k = int(reranker_cfg.get("reranker_k", 10))
        cross_encoder_model = reranker_cfg.get(
            "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        passages = [text for (_doc_id, text) in docs_to_rerank]
        if passages:
            ce_reranker = CrossEncoderReranker(cross_encoder_model)
            scores = ce_reranker.score(query, passages)
            scored = list(zip(scores, [all_docs[doc_id] for (doc_id, _text) in docs_to_rerank]))
            reranked_chunks = sorted(scored, key=lambda x: x[0], reverse=True)[:reranker_k]
            final_chunks = reranked_chunks
        else:
            final_chunks = []
    else:
        fallback_top_k = int(cfg.get("retrieval", {}).get("top_k", 20))
        final_chunks = [(1.0, all_docs[doc_id]) for doc_id in fused_doc_ids[:fallback_top_k]]

    return final_chunks


# Globals
cfg = yaml.safe_load(open("config.yaml"))
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

index, meta = load_index(cfg)
voyage_client = VoyageClient(
    os.getenv("VOYAGE_API_KEY"), model=cfg.get("embedding", {}).get("model", "voyage-context-3")
)
