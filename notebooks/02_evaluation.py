# Databricks notebook source
# COMMAND ----------
"""
Evaluation notebook

Runs a small evaluation using existing eval utilities.
"""

from pathlib import Path
import sys
import yaml


def _ensure_project_root_on_path() -> Path:
    candidates = []
    try:
        candidates.append(Path(__file__).resolve())
    except NameError:
        pass
    candidates.append(Path.cwd())

    for base in candidates:
        p = base
        for _ in range(6):
            if (
                (p / "pyproject.toml").exists()
                or (p / ".git").exists()
                or ((p / "src").exists() and (p / "apps").exists())
            ):
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))
                return p
            if p.parent == p:
                break
            p = p.parent

    ws_root = Path("/Workspace/Users")
    if ws_root.exists():
        for user_dir in ws_root.iterdir():
            if not user_dir.is_dir():
                continue
            for repo_dir in user_dir.iterdir():
                if not repo_dir.is_dir():
                    continue
                if (repo_dir / "pyproject.toml").exists() and (repo_dir / "eval").exists():
                    if str(repo_dir) not in sys.path:
                        sys.path.insert(0, str(repo_dir))
                    return repo_dir

    p = candidates[0].parent if candidates else Path.cwd()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


PROJECT_ROOT = _ensure_project_root_on_path()
cfg_path = PROJECT_ROOT / "config.yaml"
cfg = yaml.safe_load(open(cfg_path, "r"))
print("Loaded config from:", cfg_path)

# COMMAND ----------
# Minimal evaluation stub: demonstrates calling feedback metrics  # noqa: E402
from eval.feedback import groundedness_score, relevance_score  # noqa: E402

source = "ColBERT is a late interaction neural retrieval model."
statement = "ColBERT is a neural reranking model."
g = groundedness_score(source, statement)
print("Groundedness:", g)

q = "What is ColBERT?"
text = source
r = relevance_score(q, text)
print("Relevance:", r)

# COMMAND ----------
