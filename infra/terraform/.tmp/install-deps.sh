#!/bin/bash
set -euxo pipefail
/databricks/python/bin/pip install --disable-pip-version-check -q \
  faiss-cpu==1.8.0 \
  voyageai>=0.2.1 \
  rank-bm25>=0.2.2 \
  transformers==4.36.0 \
  sentence-transformers>=2.2.2 \
  torch==2.2.2 \
  tqdm>=4.66.2 \
  python-dotenv>=1.0.1 \
  pyyaml>=6.0.1 \
  colbert-ai>=0.2.19 \
  flask>=3.0.0
