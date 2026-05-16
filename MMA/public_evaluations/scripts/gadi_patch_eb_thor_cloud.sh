#!/usr/bin/env bash
# Patch EmbodiedBench thor_env.py: force CloudRendering + start_unity=False in __init__
# (ThorEnv calls self.start() afterward; ai2thor 5 must not start twice).
set -eo pipefail

EB_ROOT="${1:-${EB_ROOT:-/g/data/mv44/${USER}/EmbodiedBench}}"
THOR_PY="${EB_ROOT}/embodiedbench/envs/eb_alfred/env/thor_env.py"

if [[ ! -f "${THOR_PY}" ]]; then
  echo "ERROR: not found: ${THOR_PY}" >&2
  exit 1
fi

bak="${THOR_PY}.bak.$(date +%Y%m%d%H%M%S)"
cp -a "${THOR_PY}" "${bak}"
echo "Backup: ${bak}"

python - "${THOR_PY}" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8").read()
orig = text
changed = False

# Upgrade existing Gadi patch: add start_unity=False
if "platform=_EBCloudRendering" in text and "start_unity=False" not in text:
    text, n = re.subn(
        r"super\(\)\.__init__\(quality=quality,\s*platform=_EBCloudRendering\)",
        "super().__init__(quality=quality, platform=_EBCloudRendering, start_unity=False)",
        text,
        count=1,
    )
    if n:
        changed = True
        print("Added start_unity=False to existing CloudRendering __init__")

if "EMBODIEDBENCH_THOR_PLATFORM" not in text:
    needle = r"(\s+)super\(\)\.__init__\(quality=quality\)"
    if not re.search(needle, text):
        print("ERROR: expected super().__init__(quality=quality) in", path, file=sys.stderr)
        sys.exit(1)
    block = r'''\1import os as _eb_os
\1_thor_plat = _eb_os.environ.get("EMBODIEDBENCH_THOR_PLATFORM", "").strip()
\1if _thor_plat == "CloudRendering":
\1    try:
\1        from ai2thor.platform import CloudRendering as _EBCloudRendering
\1    except ImportError as _e:
\1        raise ImportError(
\1            "CloudRendering requires ai2thor>=5; pip install -U 'ai2thor>=5.0'"
\1        ) from _e
\1    super().__init__(quality=quality, platform=_EBCloudRendering, start_unity=False)
\1else:
\1    super().__init__(quality=quality)'''
    text, n = re.subn(needle, block, text, count=1)
    if n != 1:
        print("ERROR: patch failed", path, file=sys.stderr)
        sys.exit(1)
    changed = True
    print("Inserted EMBODIEDBENCH_THOR_PLATFORM block")

if not changed:
    if "start_unity=False" in text and "EMBODIEDBENCH_THOR_PLATFORM" in text:
        print("Already fully patched:", path)
        sys.exit(0)
    print("ERROR: no changes applied", file=sys.stderr)
    sys.exit(1)

open(path, "w", encoding="utf-8").write(text)
print("Wrote", path)
PY

echo "After patch:"
grep -n 'EMBODIEDBENCH_THOR_PLATFORM\|CloudRendering\|start_unity\|super().__init__' "${THOR_PY}" | head -14
