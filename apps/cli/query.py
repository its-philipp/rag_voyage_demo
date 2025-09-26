import argparse
from pathlib import Path
import sys
import yaml

# Ensure project root on sys.path for Databricks jobs
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.pipeline import query_system  # noqa: E402


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
