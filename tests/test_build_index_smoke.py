import os
import numpy as np
import faiss
from build_index import build_faiss


def test_build_faiss_flat_smoke(tmp_path):
    vecs = np.random.rand(100, 8).astype("float32")
    idx_dir = tmp_path / "idx"
    index = build_faiss(str(idx_dir), vecs, {"type": "flat"})
    assert isinstance(index, faiss.Index)
    assert os.path.exists(os.path.join(idx_dir, "voyage.faiss"))
