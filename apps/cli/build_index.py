from pathlib import Path
import sys

# Ensure project root on sys.path for Databricks jobs
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.index_build import main  # noqa: E402

if __name__ == "__main__":
    main()
