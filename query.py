import os, json, yaml, numpy as np, faiss
from typing import List, Dict
from src.voyage_client import VoyageClient
from src.colbert_reranker import ColBERTReranker
from src.cross_encoder_reranker import CrossEncoderReranker

def load_index(index_dir):
    index = faiss.read_index(os.path.join(index_dir, "voyage.faiss"))
    vecs = np.load(os.path.join(index_dir, "vectors.npy"))
    meta = [json.loads(l) for l in open(os.path.join(index_dir, "meta.jsonl"))]
    return index, vecs.shape[1], meta

def ann_search(index, qvec, top_m):
    # ensure normalized for IP
    faiss.normalize_L2(qvec)
    scores, ids = index.search(qvec, top_m)
    return scores[0], ids[0]

def query_system(user_query: str, cfg: Dict):
    api_key = os.getenv(cfg["embedding"]["api_key_env"], "")
    vc = VoyageClient(api_key, model=cfg["embedding"]["model"])
    index, dim, meta = load_index(cfg["index_dir"])
    # embed query
    qvec = vc.embed([user_query]).astype("float32")
    # ANN search
    scores, ids = ann_search(index, qvec, cfg["retrieval"]["top_m"])
    candidates = [meta[i] for i in ids]
    passages = [f"{c['title']}\n{c['text']}".strip() for c in candidates]

    # rerank
    if cfg["reranker"]["type"] == "colbert":
        reranker = ColBERTReranker(model_name=cfg["reranker"]["colbert_model"],
                                   max_query_len=cfg["reranker"]["max_query_len"],
                                   max_doc_len=cfg["reranker"]["max_doc_len"])
    else:
        reranker = CrossEncoderReranker()

    rerank_scores = reranker.score(user_query, passages)
    ranked = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)
    top_k = ranked[:cfg["retrieval"]["top_k"]]
    return top_k

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config.yaml"))
    results = query_system("How does ColBERT compare to cross-encoders?", cfg)
    for s, c in results:
        print(f"{s:.3f}\t{c['doc_id']}\t{c['text'][:120]}...")
