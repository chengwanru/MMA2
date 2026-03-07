"""
Root conftest.py for MMA2 unit tests.

Stubs out the mma package __init__ (which loads Flask, OpenAI, etc.) so that
isolated speculative_memory tests can import mma.speculative_memory.* without
requiring a full MMA installation.
"""

import sys
import types
import pathlib

_root = pathlib.Path(__file__).parent  # MMA2/
_mma_pkg_dir = _root / "MMA" / "MMA"  # MMA2/MMA/MMA/ = the mma package

# Make `import mma.*` resolve to MMA2/MMA/MMA/
_mma_parent = str(_root / "MMA")
if _mma_parent not in sys.path:
    sys.path.insert(0, _mma_parent)

# Prevent mma/__init__.py (which imports Flask/OpenAI) from running by
# registering a lightweight stub package before pytest loads any test modules.
if "mma" not in sys.modules:
    stub = types.ModuleType("mma")
    stub.__path__ = [str(_mma_pkg_dir)]
    stub.__package__ = "mma"
    sys.modules["mma"] = stub
