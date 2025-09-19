import os
import json
import yaml
import numpy as np
import faiss
import logging
import time
from typing import Dict
from src.voyage_client import VoyageClient
from src.colbert_reranker import ColBERTReranker
from src.cross_encoder_reranker import CrossEncoderReranker


def load_index(index_dir):
    logger = logging.getLogger(__name__)
    index_path = os.path.join(index_dir, "voyage.faiss")
    vectors_path = os.path.join(index_dir, "vectors.npy")
    meta_path = os.path.join(index_dir, "meta.jsonl")
    index = faiss.read_index(index_path)
    vecs = np.load(vectors_path)
    with open(meta_path, "r") as fh:
        meta = [json.loads(line) for line in fh]
    logger.info(
        "Loaded index from %s with dim=%s and %s metadata rows",
        index_path,
        vecs.shape[1],
        len(meta),
    )
    return index, vecs.shape[1], meta


def ann_search(index, qvec, top_m):
    # ensure normalized for IP
    faiss.normalize_L2(qvec)
    scores, ids = index.search(qvec, top_m)
    return scores[0], ids[0]


def query_system(user_query: str, cfg: Dict):
    logger = logging.getLogger(__name__)
    logger.info("Query start: '%s'", user_query)

    api_key = os.getenv(cfg["embedding"]["api_key_env"], "")
    vc = VoyageClient(api_key, model=cfg["embedding"]["model"])
    index, dim, meta = load_index(cfg["index_dir"])

    t0 = time.perf_counter()
    qvec = vc.embed([user_query]).astype("float32")
    t_embed_ms = (time.perf_counter() - t0) * 1000
    logger.info("Embedded query in %.1f ms", t_embed_ms)

    t1 = time.perf_counter()
    scores, ids = ann_search(index, qvec, cfg["retrieval"]["top_m"])
    t_retr_ms = (time.perf_counter() - t1) * 1000
    logger.info(
        "ANN search done in %.1f ms (top_m=%s)", t_retr_ms, cfg["retrieval"]["top_m"]
    )

    candidates = [meta[i] for i in ids]
    passages = [f"{c['title']}\n{c['text']}".strip() for c in candidates]

    # rerank
    if cfg["reranker"]["type"] == "colbert":
        reranker = ColBERTReranker(
            model_name=cfg["reranker"]["colbert_model"],
            max_query_len=cfg["reranker"]["max_query_len"],
            max_doc_len=cfg["reranker"]["max_doc_len"],
        )
    else:
        reranker = CrossEncoderReranker()

    t2 = time.perf_counter()
    rerank_scores = reranker.score(user_query, passages)
    t_rerank_ms = (time.perf_counter() - t2) * 1000
    logger.info("Reranked %s passages in %.1f ms", len(passages), t_rerank_ms)

    ranked = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)
    top_k = ranked[: cfg["retrieval"]["top_k"]]
    logger.info("Returning top_k=%s results", cfg["retrieval"]["top_k"])
    return top_k


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    cfg = yaml.safe_load(open("config.yaml"))
    results = query_system("How does ColBERT compare to cross-encoders?", cfg)
    for s, c in results:
        print(f"{s:.3f}\t{c['doc_id']}\t{c['text'][:120]}...")
