#!/usr/bin/env bash
# One-time fix on Gadi (login OK) for headless Thor on PBS.
set -eo pipefail

EB_ROOT="${1:-${EB_ROOT:-/g/data/mv44/${USER}/EmbodiedBench}}"
EB_PY="${EB_ROOT}/embodiedbench/envs/eb_alfred/EBAlfEnv.py"

if [[ ! -f "${EB_PY}" ]]; then
  echo "ERROR: not found: ${EB_PY}" >&2
  exit 1
fi

bak="${EB_PY}.bak.$(date +%Y%m%d%H%M%S)"
cp -a "${EB_PY}" "${bak}"
echo "Backup: ${bak}"

python - "${EB_PY}" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8").read()
orig = text
changed = False

xdisplay_helper = '''def _eb_x_display_from_env():
    """Ignore PBS defaults (:0.0, 1); None enables CloudRendering when platform is set."""
    v = os.environ.get("X_DISPLAY")
    if v is None:
        return None
    v = str(v).strip()
    if v in ("", ":0.0", ":0", "0.0", "1"):
        return None
    return v


X_DISPLAY = _eb_x_display_from_env()'''

pop_block = """        if X_DISPLAY is None:
            os.environ.pop("DISPLAY", None)
            os.environ.pop("X_DISPLAY", None)"""

# 1) Module-level X_DISPLAY helper
if "_eb_x_display_from_env" not in text:
    text2, n1 = re.subn(
        r"^X_DISPLAY\s*=\s*os\.environ\.get\([^\n]+\)\s*(#.*)?$",
        xdisplay_helper,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n1 == 0:
        text2, n1 = re.subn(
            r"^X_DISPLAY\s*=\s*['\"]1['\"]\s*$",
            xdisplay_helper,
            text,
            count=1,
            flags=re.MULTILINE,
        )
    if n1:
        text = text2
        changed = True
        print("Patched X_DISPLAY helper")

# 2) Deduplicate pop blocks and ensure exactly one block before ThorConnector
thor_needle = r"self\.env = ThorConnector\(x_display=X_DISPLAY"
if re.search(thor_needle, text):
  text2, n2 = re.subn(
      r"(?:[ \t]*if X_DISPLAY is None:\n[ \t]*os\.environ\.pop\(\"DISPLAY\", None\)\n(?:[ \t]*os\.environ\.pop\(\"X_DISPLAY\", None\)\n)?)+"
      r"([ \t]*self\.env = ThorConnector\(x_display=X_DISPLAY)",
      pop_block + r"\n\1",
      text,
      count=1,
  )
  if n2:
      if text2 != text:
          changed = True
          print("Normalized ThorConnector pop DISPLAY block")
      text = text2
  elif pop_block not in text:
      text2, n3 = re.subn(
          r"(\n)([ \t]*)(self\.env = ThorConnector\(x_display=X_DISPLAY)",
          "\n" + pop_block + r"\n\2\3",
          text,
          count=1,
      )
      if n3:
          text = text2
          changed = True
          print("Inserted ThorConnector pop DISPLAY block")

if not changed and "_eb_x_display_from_env" in text and pop_block in text:
    print("Already fully patched:", path)
    sys.exit(0)

if text == orig:
    print("ERROR: no changes applied; inspect:", path, file=sys.stderr)
    sys.exit(1)

if "import os" not in text:
    text = "import os\n" + text
elif not re.search(r"^import os\b|^from os\b", text, re.MULTILINE):
    text = "import os\n" + text

open(path, "w", encoding="utf-8").write(text)
print("Wrote", path)
PY

echo "After patch:"
grep -n -E 'X_DISPLAY|pop\("DISPLAY"|ThorConnector' "${EB_PY}" | head -12
