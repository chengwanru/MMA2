"""Path helpers for MMA/public_evaluations benchmark layout."""

from __future__ import annotations

import sys
from pathlib import Path

_PEV_ROOT = Path(__file__).resolve().parent.parent


def pev_root() -> Path:
    return _PEV_ROOT


def bench_dir(name: str) -> Path:
    return _PEV_ROOT / name


def ensure_pev_on_syspath() -> Path:
    root = str(_PEV_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return _PEV_ROOT


def ensure_mma_package() -> Path:
    """Insert MMA/ (parent of public_evaluations) so `import mma` works."""
    mma_parent = _PEV_ROOT.parent
    for base in (mma_parent, _PEV_ROOT, Path.cwd(), Path.cwd().parent):
        if (base / "mma" / "__init__.py").exists() or (base / "MMA" / "__init__.py").exists():
            s = str(base.resolve())
            if s not in sys.path:
                sys.path.insert(0, s)
            return base
    raise RuntimeError(
        "Could not locate mma package. Run from MMA2/MMA or set PYTHONPATH to MMA/."
    )
