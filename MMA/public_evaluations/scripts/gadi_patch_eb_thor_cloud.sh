#!/usr/bin/env bash
# Patch EmbodiedBench thor_env.py to force ai2thor CloudRendering on Gadi (login OK).
set -eo pipefail

EB_ROOT="${1:-${EB_ROOT:-/g/data/mv44/${USER}/EmbodiedBench}}"
THOR_PY="${EB_ROOT}/embodiedbench/envs/eb_alfred/env/thor_env.py"

if [[ ! -f "${THOR_PY}" ]]; then
  echo "ERROR: not found: ${THOR_PY}" >&2
  exit 1
fi

if grep -q 'EMBODIEDBENCH_THOR_PLATFORM' "${THOR_PY}" 2>/dev/null; then
  echo "Already patched: ${THOR_PY}"
  grep -n 'EMBODIEDBENCH_THOR_PLATFORM\|CloudRendering' "${THOR_PY}" | head -8
  exit 0
fi

bak="${THOR_PY}.bak.$(date +%Y%m%d%H%M%S)"
cp -a "${THOR_PY}" "${bak}"
echo "Backup: ${bak}"

python - "${THOR_PY}" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8").read()

needle = r"(\s+)super\(\)\.__init__\(quality=quality\)"
if not re.search(needle, text):
    print("ERROR: expected super().__init__(quality=quality) in", path, file=sys.stderr)
    sys.exit(1)

block = r'''\1import os as _eb_os
\1_thor_plat = _eb_os.environ.get("EMBODIEDBENCH_THOR_PLATFORM", "").strip()
\1if _thor_plat == "CloudRendering":
\1    from ai2thor.platform import CloudRendering as _EBCloudRendering
\1    super().__init__(quality=quality, platform=_EBCloudRendering)
\1else:
\1    super().__init__(quality=quality)'''

text, n = re.subn(needle, block, text, count=1)
if n != 1:
    print("ERROR: patch failed", path, file=sys.stderr)
    sys.exit(1)

open(path, "w", encoding="utf-8").write(text)
print("Patched", path)
PY

echo "After patch:"
grep -n 'EMBODIEDBENCH_THOR_PLATFORM\|CloudRendering\|super().__init__' "${THOR_PY}" | head -12
