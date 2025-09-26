import argparse
from pathlib import Path
import sys
import yaml


def _ensure_project_root_on_path() -> None:
    candidates = []
    try:
        candidates.append(Path(__file__).resolve())
    except NameError:
        pass
    candidates.append(Path.cwd())

    for base in candidates:
        p = base
        for _ in range(6):
            if (p / "pyproject.toml").exists() or (p / ".git").exists():
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))
                return
            if p.parent == p:
                break
            p = p.parent


_ensure_project_root_on_path()

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
