#!/usr/bin/env python3
"""
Patch EmbodiedBench thor_env.py to remap ALFRED object_poses hashes before SetObjectPoses.

Usage:
  python patch_remap_object_poses.py
  python patch_remap_object_poses.py /path/to/EmbodiedBench/embodiedbench/envs/eb_alfred/env/thor_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

MARKER = "def _remap_alfred_object_poses("

HELPER = '''
def _remap_alfred_object_poses(object_poses, objects):
    """
    ALFRED JSON uses frozen instance names (e.g. Ladle_f4537974). ai2thor 5 /
    CloudRendering spawns new hashes on each reset (e.g. Ladle_381cdb86).
    Map each pose to the closest same-type object in the current scene before
    SetObjectPoses.
    """
    if not object_poses or not objects:
        return object_poses

    from collections import defaultdict
    import math

    def _xyz(p):
        return (float(p["x"]), float(p["y"]), float(p["z"]))

    by_type = defaultdict(list)
    for obj in objects:
        t = obj.get("objectType") or ""
        if t:
            by_type[t].append(obj)

    used = set()
    remapped = []
    for pose in object_poses:
        old_name = pose.get("objectName") or ""
        if not old_name:
            continue
        obj_type = old_name.split("_", 1)[0]
        cands = [o for o in by_type.get(obj_type, []) if o.get("name") not in used]
        if not cands:
            continue
        pos = pose.get("position") or {}
        if all(k in pos for k in ("x", "y", "z")):
            target = _xyz(pos)
            chosen = min(cands, key=lambda o: math.dist(_xyz(o["position"]), target))
        else:
            chosen = cands[0]
        used.add(chosen["name"])
        remapped.append({**pose, "objectName": chosen["name"]})
    return remapped if remapped else object_poses

'''

OLD_CALL = "        super().step((dict(action='SetObjectPoses', objectPoses=object_poses)))"
NEW_CALL = """        scene_objects = self.last_event.metadata.get("objects") or []
        poses = _remap_alfred_object_poses(object_poses, scene_objects)
        super().step((dict(action='SetObjectPoses', objectPoses=poses)))"""


def patch_thor_env(text: str) -> tuple[str, bool]:
    if MARKER in text:
        return text, False
    anchor = "class ThorEnv(Controller):"
    idx = text.find(anchor)
    if idx < 0:
        return text, False
    text = text[:idx] + HELPER + text[idx:]
    if OLD_CALL not in text:
        return text, False
    text = text.replace(OLD_CALL, NEW_CALL, 1)
    return text, True


def main() -> int:
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("embodiedbench/envs/eb_alfred/env/thor_env.py")
    )
    if not path.is_file():
        print(f"ERROR: not found: {path}", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"Already patched: {path}")
        return 0
    new_text, ok = patch_thor_env(text)
    if not ok:
        print(f"ERROR: could not patch {path}", file=sys.stderr)
        return 1
    path.write_text(new_text, encoding="utf-8")
    print(f"Patched OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
