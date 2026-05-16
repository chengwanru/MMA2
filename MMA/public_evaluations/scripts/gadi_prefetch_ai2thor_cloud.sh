#!/usr/bin/env bash
# Download ai2thor CloudRendering build on login (needs internet). Compute nodes are offline.
# Usage:
#   conda activate /g/data/mv44/$USER/envs/embench
#   bash gadi_check_ai2thor.sh          # must pass
#   bash gadi_prefetch_ai2thor_cloud.sh
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

export EMBODIEDBENCH_THOR_PLATFORM=CloudRendering
python - <<'PY'
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

print("Prefetching CloudRendering build (downloads to ~/.ai2thor on first run) ...")
c = Controller(platform=CloudRendering, width=300, height=300, start_unity=False)
print("OK:", type(c), "platform=", getattr(c, "_build", None) and c._build.platform)
PY

echo "Done. Check: ls ${AI2THOR_HOME}/releases | head"
