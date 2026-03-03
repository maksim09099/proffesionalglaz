from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts"


def resolve_project_root(start: Path | None = None) -> Path:
    """Resolve project root for scripts/notebooks in PyCharm.

    Looks upward for .git and falls back to package-derived PROJECT_ROOT.
    """
    cursor = (start or Path.cwd()).resolve()
    for candidate in [cursor, *cursor.parents]:
        if (candidate / ".git").exists():
            return candidate
    return PROJECT_ROOT
