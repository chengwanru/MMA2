# EmbodiedBench upstream patches (MMA integration)

Apply these **inside your EmbodiedBench clone** (same paths as GitHub `EmbodiedBench/EmbodiedBench`).

```bash
cd /path/to/EmbodiedBench
bash /path/to/MMA2/MMA/public_evaluations/patches/embodiedbench_upstream/apply_patches.sh
```

## What each patch does

| Patch | Purpose |
|-------|---------|
| `001_EBAlfEnv_invalid_jsonl.patch` | When `EMBODIEDBENCH_INVALID_LOG_JSONL` is set, append one JSON line per **failed** Thor interaction with `reason_code` (not_visible, not_reachable, …). |
| `002_custom_model_feedback.patch` | Forward optional form fields (e.g. `last_env_feedback`) on POST to the custom model server. |
| `003_vlm_planner_feedback.patch` | After each step, pass previous `env_feedback` to `CustomModel.respond` as `last_env_feedback` (MMA server reads this), and heuristically attach `reason_code` (`not_visible` / `not_reachable` / `collision` / `blocked`) when text matches. |
| `004_EBAlfEnv_x_display_env.patch` | Replace hardcoded `X_DISPLAY = '1'` with `os.environ.get("X_DISPLAY")` so Gadi smoke can use CloudRendering (unset) instead of `:1` / xdpyinfo. |

## Environment variables (client / eval job)

- `EMBODIEDBENCH_INVALID_LOG_JSONL=/path/to/invalid_reason.jsonl` — enable JSONL invalid logging (patch 001).
- Existing `server_url` for custom backend unchanged.

## Revert

```bash
cd /path/to/EmbodiedBench
patch -R -p1 < /path/to/001_EBAlfEnv_invalid_jsonl.patch
# ... etc.
```
