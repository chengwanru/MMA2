# EmbodiedBench upstream patches (MMA integration)

Apply these **inside your EmbodiedBench clone** (same paths as GitHub `EmbodiedBench/EmbodiedBench`).

```bash
cd /path/to/EmbodiedBench
bash /path/to/MMA2/MMA/public_evaluations/embodiedbench/patches/embodiedbench_upstream/apply_patches.sh
```

## What each patch does

| Patch | Purpose |
|-------|---------|
| `001_EBAlfEnv_invalid_jsonl.patch` | When `EMBODIEDBENCH_INVALID_LOG_JSONL` is set, append one JSON line per **failed** Thor interaction with `reason_code` (not_visible, not_reachable, …). |
| `002_custom_model_feedback.patch` | Forward optional form fields (e.g. `last_env_feedback`) on POST to the custom model server. |
| `003_vlm_planner_feedback.patch` | After each step, pass previous `env_feedback` to `CustomModel.respond` as `last_env_feedback` (MMA server reads this), and heuristically attach `reason_code` (`not_visible` / `not_reachable` / `collision` / `blocked`) when text matches. |
| `004_EBAlfEnv_x_display_env.patch` | Replace hardcoded `X_DISPLAY = '1'` with `os.environ.get("X_DISPLAY")` so Gadi smoke can use CloudRendering (unset) instead of `:1` / xdpyinfo. |
| `005_EBAlfEnv_unset_display_for_cloud.patch` | Before `ThorConnector`, `pop("DISPLAY")` when `X_DISPLAY is None` (PBS often sets `DISPLAY=:0.0`, which ai2thor would otherwise validate). |
| `006_vlm_planner_instruction.patch` | Pass Thor `episode_language_instruction` to custom model POST as form field `instruction` (authoritative task text; not buried in n-shot prompt). If `patch -p1` fails (upstream drift), run `python3 patch_vlm_instruction.py` from this directory with EmbodiedBench as cwd, or use `apply_patches.sh` which falls back automatically. |
| `007_thor_remap_object_poses.patch` | **LTU / ai2thor 5 fix:** ALFRED JSON uses old object instance hashes (`Ladle_f4537974`) but CloudRendering spawns new hashes (`Ladle_381cdb86`). Remap poses by object type + 3D position before `SetObjectPoses` so restore succeeds. Fallback: `python3 patch_remap_object_poses.py`. |

## Environment variables (client / eval job)

- `EMBODIEDBENCH_INVALID_LOG_JSONL=/path/to/invalid_reason.jsonl` — enable JSONL invalid logging (patch 001).
- Existing `server_url` for custom backend unchanged.

## Revert

```bash
cd /path/to/EmbodiedBench
patch -R -p1 < /path/to/001_EBAlfEnv_invalid_jsonl.patch
# ... etc.
```
