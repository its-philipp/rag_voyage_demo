# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install -q pyyaml voyageai openai>=1.0.0 typing_extensions>=4.7 python-dotenv
# COMMAND ----------
# Restart Python so newly installed libraries are used in this session
try:
    dbu = globals().get("dbutils")
    if dbu is not None:
        dbu.library.restartPython()
except Exception as e:  # noqa: BLE001
    print("If restart is unavailable, use the blue 'Restart Python' banner or rerun this cell:", e)
# COMMAND ----------
"""
Load VOYAGE_API_KEY from Databricks secret scope if not present.
"""
import os  # noqa: E402

if not os.getenv("VOYAGE_API_KEY"):
    try:
        dbu = globals().get("dbutils")
        if dbu is not None:
            os.environ["VOYAGE_API_KEY"] = dbu.secrets.get(
                scope="rag-voyage-demo", key="VOYAGE_API_KEY"
            )
            print("VOYAGE_API_KEY set from secret scope.")
    except Exception as e:  # noqa: BLE001
        print("Warning: Could not load VOYAGE_API_KEY from secret scope:", e)
# Also try to load OpenAI key if missing
if not os.getenv("OPENAI_API_KEY"):
    try:
        dbu = globals().get("dbutils")
        if dbu is not None:
            os.environ["OPENAI_API_KEY"] = dbu.secrets.get(
                scope="rag-voyage-demo", key="OPENAI_API_KEY"
            )
            print("OPENAI_API_KEY set from secret scope.")
    except Exception as e:  # noqa: BLE001
        print("Warning: Could not load OPENAI_API_KEY from secret scope:", e)
# COMMAND ----------
# COMMAND ----------
"""
Evaluation notebook

Runs a small evaluation using existing eval utilities.
"""

from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import yaml  # noqa: E402


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
