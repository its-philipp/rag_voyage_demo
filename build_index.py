import os
import json
import yaml
import numpy as np
import faiss
import logging
import time
from tqdm import tqdm
from src.voyage_client import VoyageClient
from src.chunking import make_chunks_with_context


def load_docs(path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def build_faiss(index_dir, vecs, index_cfg):
    logger = logging.getLogger(__name__)
    os.makedirs(index_dir, exist_ok=True)
    d = vecs.shape[1]
    index_type = index_cfg.get("type", "ivf_pq")
    if index_type == "flat":
        index = faiss.IndexFlatIP(d)
        index.add(vecs)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(d, 32)
        faiss.ParameterSpace().set_index_parameter(index, "efSearch", 128)
        faiss.ParameterSpace().set_index_parameter(index, "efConstruction", 200)
        index.hnsw.efSearch = 128
        index.add(vecs)
    elif index_type == "ivf_pq":
        nlist = int(index_cfg.get("nlist", 4096))
        m = int(index_cfg.get("m", 32))
        quantizer = faiss.IndexFlatIP(d)
        # FAISS requires a larger number of training points than nlist for clustering.
        # Empirically require at least 4 * nlist training points; otherwise fall back
        # to a flat index to avoid errors like "Number of training points (N) should
        # be at least as large as number of clusters (k)" when N < k.
        min_train_needed = int(nlist * 4)
        if len(vecs) < min_train_needed:
            print(
                f"Warning: only {len(vecs)} vectors available but IVFPQ with nlist={nlist} requires at least {min_train_needed} training points;"
                " falling back to flat index to avoid FAISS training error."
            )
            index = faiss.IndexFlatIP(d)
            faiss.normalize_L2(vecs)
            index.add(vecs)
            faiss.write_index(index, os.path.join(index_dir, "voyage.faiss"))
            return index
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = int(index_cfg.get("nprobe", 16))
        # train
        train_samples = vecs[
            np.random.choice(len(vecs), min(20000, len(vecs)), replace=False)
        ]
        faiss.normalize_L2(train_samples)
        index.train(train_samples)
        # add
        faiss.normalize_L2(vecs)
        index.add(vecs)
    else:
        raise ValueError("Unknown faiss.type")
    index_path = os.path.join(index_dir, "voyage.faiss")
    faiss.write_index(index, index_path)
    logger.info(
        "FAISS index written to %s (type=%s, dim=%s, size=%s)",
        index_path,
        index_type,
        d,
        len(vecs),
    )
    return index


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    # Load config relative to this script so running the script from a
    # different working directory still finds the config file.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. Run this script from the project folder or create a config.yaml there."
        )
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    api_key = os.getenv(cfg["embedding"]["api_key_env"], "")
    vc = VoyageClient(api_key, model=cfg["embedding"]["model"])

    # Resolve data path relative to the script directory if necessary
    data_path = cfg.get("data_path")
    if not data_path:
        raise ValueError("config.yaml missing 'data_path' entry")
    if not os.path.isabs(data_path):
        data_path = os.path.join(script_dir, data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data file not found: {data_path}")
    docs = load_docs(data_path)
    # chunk
    all_chunks = []
    for d in docs:
        all_chunks.extend(make_chunks_with_context(d, max_sentences=3, overlap=1))

    # embed (concatenate context + text)
    texts = [f"{c['title']}\n{c['text']}".strip() for c in all_chunks]
    if len(texts) == 0:
        print("No chunks to embed; exiting.")
        return
    # batch embedding requests to avoid exceeding provider limits
    conf_bs = int(cfg.get("embedding", {}).get("batch_size", 16))
    # cap per-request size to provider hard limit (Voyage complained at 1000)
    per_request = min(conf_bs, 1000)
    batches = [texts[i : i + per_request] for i in range(0, len(texts), per_request)]
    vecs_parts = []
    t0 = time.perf_counter()
    for b in tqdm(batches, desc="Embedding batches"):
        part = vc.embed(b).astype("float32")
        vecs_parts.append(part)
    vecs = np.vstack(vecs_parts)
    embed_ms = (time.perf_counter() - t0) * 1000
    logging.getLogger(__name__).info(
        "Embedded %s chunks in %.1f ms", len(texts), embed_ms
    )
    # (optional) store metadata
    # Resolve index_dir relative to script dir
    index_dir = cfg.get("index_dir", "index")
    if not os.path.isabs(index_dir):
        index_dir = os.path.join(script_dir, index_dir)
    os.makedirs(index_dir, exist_ok=True)
    np.save(os.path.join(index_dir, "vectors.npy"), vecs)
    with open(os.path.join(index_dir, "meta.jsonl"), "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")

    # build faiss
    build_faiss(index_dir, vecs, cfg.get("faiss", {}))
    print("Index built:", index_dir, "num_chks=", len(all_chunks))


if __name__ == "__main__":
    main()
