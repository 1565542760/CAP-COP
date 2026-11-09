# Final-method dependency subset; retained scientific definitions are source-verbatim.
from __future__ import annotations

import os

from pathlib import Path

def project_root() -> Path:
    return Path(os.environ.get("CAP_COP_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

def resolve_project_path(relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"Absolute paths are not allowed in public configs: {path}")
    return project_root() / path
