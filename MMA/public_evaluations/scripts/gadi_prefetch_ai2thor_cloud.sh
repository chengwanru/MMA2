#!/usr/bin/env bash
# Download ai2thor CloudRendering build on login (needs internet). Compute nodes are offline.
# Usage on Gadi login:
#   export CONDA_ACTIVATE_SCRIPT=...  # or: source .../conda.sh && conda activate .../embench
#   bash gadi_prefetch_ai2thor_cloud.sh
set -eo pipefail

AI2THOR_HOME="${AI2THOR_HOME:-/g/data/mv44/${USER}/ai2thor}"
mkdir -p "${AI2THOR_HOME}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate embench first, e.g.: conda activate /g/data/mv44/\$USER/envs/embench" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# Thor caches under ~/.ai2thor; symlink so compute jobs reuse gdata copy.
if [[ ! -e "${HOME}/.ai2thor" ]]; then
  ln -sfn "${AI2THOR_HOME}" "${HOME}/.ai2thor"
  echo "Linked ${HOME}/.ai2thor -> ${AI2THOR_HOME}"
fi

export EMBODIEDBENCH_THOR_PLATFORM=CloudRendering
python - <<'PY'
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

print("Prefetching CloudRendering build (first run downloads to ~/.ai2thor) ...")
c = Controller(platform=CloudRendering, width=300, height=300, start_unity=False)
print("OK:", c)
PY

echo "Done. CloudRendering binaries should be under ${AI2THOR_HOME}/releases"
