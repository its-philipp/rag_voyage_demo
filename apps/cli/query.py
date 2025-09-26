import argparse
from pathlib import Path
import sys
import yaml


def _ensure_project_root_on_path() -> None:
    """Add project root to sys.path for imports."""

    # Try multiple strategies to find the project root
    candidates = []

    # Strategy 1: Use __file__ if available
    try:
        candidates.append(Path(__file__).resolve().parent.parent.parent)
    except NameError:
        pass

    # Strategy 2: Check current working directory and its parents
    cwd = Path.cwd()
    candidates.append(cwd)

    # Strategy 3: Check if we're in Databricks workspace
    if "/Workspace/Users/" in str(cwd):
        # In Databricks, the repo is typically at /Workspace/Users/{email}/{repo_name}.git
        parts = str(cwd).split("/")
        if ".git" in parts[-1] or ".git" in parts[-2] or ".git" in parts[-3]:
            # Find the .git directory level
            for i in range(len(parts)):
                if ".git" in parts[i]:
                    repo_root = "/".join(parts[: i + 1])
                    candidates.append(Path(repo_root))
                    break

    # Strategy 4: Look for project markers going up from cwd
    p = cwd
    for _ in range(10):  # Check up to 10 levels
        candidates.append(p)
        if p.parent == p:
            break
        p = p.parent

    # Try each candidate
    for candidate in candidates:
        if candidate.exists():
            # Check if this looks like the project root
            if (
                (candidate / "pyproject.toml").exists()
                or (candidate / "src").exists()
                and (candidate / "apps").exists()
            ):
                if str(candidate) not in sys.path:
                    print(f"Adding to sys.path: {candidate}")
                    sys.path.insert(0, str(candidate))
                return

    # Last resort: if we can guess the Databricks repo path
    databricks_repo = Path("/Workspace/Users/philipptrinh@gmail.com/rag_voyage_demo.git")
    if databricks_repo.exists() and str(databricks_repo) not in sys.path:
        print(f"Using Databricks repo path: {databricks_repo}")
        sys.path.insert(0, str(databricks_repo))


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
