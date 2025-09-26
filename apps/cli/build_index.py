from pathlib import Path

from src.index_build import main  # noqa: E402

import sys  # noqa: E402

# Ensure project root on sys.path for Databricks jobs  # noqa: E402
root = Path(__file__).resolve().parents[2]  # noqa: E402
if str(root) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(root))  # noqa: E402

if __name__ == "__main__":
    main()
