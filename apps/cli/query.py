import argparse
import yaml
from src.pipeline import query_system


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
