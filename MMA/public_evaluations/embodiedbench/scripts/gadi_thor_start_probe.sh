#!/usr/bin/env bash
# Run on a GPU compute node (qsub -I or short PBS) to test Thor CloudRendering startup.
# Usage: conda activate embench && bash gadi_thor_start_probe.sh
set -eo pipefail

export EMBODIEDBENCH_THOR_PLATFORM=CloudRendering
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
if ! command -v vulkaninfo >/dev/null 2>&1; then
  echo "ERROR: vulkaninfo not in PATH. On login: conda install -c conda-forge vulkan-tools" >&2
  exit 1
fi
echo "vulkaninfo=$(command -v vulkaninfo)"
unset DISPLAY X_DISPLAY || true
ln -sfn "/g/data/mv44/${USER}/ai2thor" "${HOME}/.ai2thor" 2>/dev/null || true

echo "nvidia-smi:"; nvidia-smi || true
echo "ai2thor cache: $(du -sh "${HOME}/.ai2thor" 2>/dev/null | cut -f1)"

timeout 600 python -u - <<'PY'
import os
import signal

def _alarm(signum, frame):
    raise TimeoutError("Thor probe exceeded 600s")

signal.signal(signal.SIGALRM, _alarm)
signal.alarm(600)

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

print("Starting Controller(CloudRendering, gpu_device=0, 300x300)...", flush=True)
c = Controller(
    platform=CloudRendering,
    gpu_device=0,
    headless=True,
    width=300,
    height=300,
    start_unity=True,
)
print("OK:", c._build.platform.name(), flush=True)
signal.alarm(0)
PY

echo "probe exit=$?"
