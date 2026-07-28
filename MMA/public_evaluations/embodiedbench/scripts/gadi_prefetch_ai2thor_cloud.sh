#!/usr/bin/env bash
# Download ai2thor CloudRendering build on login (needs internet). Compute nodes are offline.
# First run can take 10–30+ min with little console output — watch cache dir grow.
#
# Usage:
#   conda activate /g/data/mv44/$USER/envs/embench
#   bash gadi_check_ai2thor.sh
#   bash gadi_prefetch_ai2thor_cloud.sh
#
# In another terminal while waiting:
#   watch -n 10 'du -sh /g/data/mv44/$USER/ai2thor 2>/dev/null; ls /g/data/mv44/$USER/ai2thor/releases 2>/dev/null | tail -3'
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI2THOR_HOME="${AI2THOR_HOME:-/g/data/mv44/${USER}/ai2thor}"
mkdir -p "${AI2THOR_HOME}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate embench first, e.g.: conda activate /g/data/mv44/\$USER/envs/embench" >&2
  exit 1
fi

if ! bash "${SCRIPT_DIR}/gadi_check_ai2thor.sh"; then
  echo "" >&2
  echo "Upgrade ai2thor on login (network required):" >&2
  echo "  pip install -U 'ai2thor>=5.0'" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
if [[ ! -e "${HOME}/.ai2thor" ]]; then
  ln -sfn "${AI2THOR_HOME}" "${HOME}/.ai2thor"
  echo "Linked ${HOME}/.ai2thor -> ${AI2THOR_HOME}"
fi

echo "Cache dir: ${AI2THOR_HOME} (before: $(du -sh "${AI2THOR_HOME}" 2>/dev/null | cut -f1 || echo 0))"
echo "Downloading CloudRendering Unity build only (download_only=True, no Unity start) ..."
echo "If this sits silent for a while, check growth: du -sh ${AI2THOR_HOME}"

export EMBODIEDBENCH_THOR_PLATFORM=CloudRendering
export PYTHONUNBUFFERED=1
python -u - <<'PY'
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

print("Calling Controller(..., platform=CloudRendering, download_only=True) ...", flush=True)
c = Controller(platform=CloudRendering, width=300, height=300, download_only=True)
plat = getattr(c, "_build", None) and c._build.platform
print("OK. platform=", plat, "base_dir=", getattr(c, "base_dir", None), flush=True)
PY

echo "After download: $(du -sh "${AI2THOR_HOME}" 2>/dev/null | cut -f1)"
echo "releases:"
ls "${AI2THOR_HOME}/releases" 2>/dev/null | head -10 || echo "(empty — download may have failed)"
echo "Done."
