import argparse
from pathlib import Path

from src.pipeline import query_system  # noqa: E402

import sys  # noqa: E402
import yaml  # noqa: E402

# Ensure project root on sys.path for Databricks jobs  # noqa: E402
root = Path(__file__).resolve().parents[2]  # noqa: E402
if str(root) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(root))  # noqa: E402


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="The query to search for.")
    args = parser.parse_args()

    results = query_system(args.query, cfg)
    for score, item in results:
        print(f"{score:.3f}\t{item['doc_id']}\t{item['text'][:120]}...")


if __name__ == "__main__":
    main()
