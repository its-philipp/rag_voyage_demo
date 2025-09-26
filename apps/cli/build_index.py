from pathlib import Path
import sys


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

from src.index_build import main  # noqa: E402

if __name__ == "__main__":
    main()
