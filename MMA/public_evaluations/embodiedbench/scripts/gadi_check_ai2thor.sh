#!/usr/bin/env bash
# Print ai2thor version / CloudRendering support (run after: conda activate .../embench).
set -eo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate embench first." >&2
  exit 1
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python - <<'PY'
import importlib.util
import sys

try:
    import ai2thor
except ImportError as e:
    print("ERROR: ai2thor not installed:", e)
    sys.exit(1)

ver = getattr(ai2thor, "__version__", "unknown")
print("ai2thor version:", ver)
print("python:", sys.executable)

has_platform = importlib.util.find_spec("ai2thor.platform") is not None
print("ai2thor.platform:", "yes" if has_platform else "NO (need pip install -U 'ai2thor>=5.0')")

if has_platform:
    from ai2thor.platform import CloudRendering
    from ai2thor.controller import Controller
    import inspect
    sig = inspect.signature(Controller.__init__)
    print("Controller(platform=...):", "platform" in sig.parameters)
    print("CloudRendering:", CloudRendering)
else:
    sys.exit(2)
PY
