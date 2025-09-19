import sys
from pathlib import Path

# Ensure project root is importable so tests can import `src` and root modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
