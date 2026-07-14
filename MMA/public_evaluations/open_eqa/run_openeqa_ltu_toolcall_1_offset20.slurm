#!/bin/bash
#SBATCH --job-name=openeqa_tc1o20
#SBATCH -p day
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o logs/openeqa_toolcall_1_offset20_%j.log
#SBATCH -e logs/openeqa_toolcall_1_offset20_%j.err

# Single-sample verify for offset-20 pipeline (ds index 20: "What color is the car?").
# Same memorize/QA config as run_openeqa_ltu_toolcall_20_offset20.slurm.
#
#   cd /data/group/zhaolab/project/MMA2/MMA/public_evaluations/open_eqa
#   git pull && mkdir -p logs
#   rm -rf /tmp/openeqa_home ~/.mma/sqlite.db*
#   sbatch run_openeqa_ltu_toolcall_1_offset20.slurm
#   tail -f logs/openeqa_toolcall_1_offset20_<jobid>.log

set -euo pipefail

export OFFSET=20
export OFFSET_EVAL=0
export LIMIT=1
export MAX_SAMPLES=1
export ALL_FRAMES=0
export FRAMES_PER_EPISODE=8
export VARIANTS=ours
export INPUT="${INPUT:-data/open-eqa-multimodal-1-o20.json}"
export OPENEQA_DEBUG=1
export OPENEQA_EPISODIC_TOOL_CALL=0
export OPENEQA_DIRECT_EPISODIC=1
export OPENEQA_SKIP_ABSORB=1
export OPENEQA_QA_MAX_TOKENS=64
export OPENEQA_ABSORB_BATCH_SIZE=1
export OPENEQA_COLLECT_SD_STATS=1
export OPENEQA_TRUST_GATE=1
export OPENEQA_VERIFY_REJECT_BAD_DRAFT=1
export MMA_MEMORY_BIAS_DEDUP=1
export MMA_MEMORY_BIAS_USE_SUMMARY=0
export MMA_MEMORY_BIAS_FILTER_INVISIBLE=1
export MMA_MEMORY_BIAS_SCALE=0.35
export MMA_MEMORY_BIAS_TOP_K=1
export MMA_REJECT_STRATEGY="${MMA_REJECT_STRATEGY:-greedy+semantic}"
export MMA_SEMANTIC_THRESHOLD="${MMA_SEMANTIC_THRESHOLD:-0.78}"
export MMA_SEMANTIC_TOP_K="${MMA_SEMANTIC_TOP_K:-8}"
export MMA_DRAFT_FAST_SINGLE_STEP="${MMA_DRAFT_FAST_SINGLE_STEP:-1}"
export MMA_SD_TARGET_KV_CACHE="${MMA_SD_TARGET_KV_CACHE:-1}"
export MMA_ENABLE_VISUAL_ROUTING="${MMA_ENABLE_VISUAL_ROUTING:-1}"
export OPENEQA_VL_MAX_PIXELS="${OPENEQA_VL_MAX_PIXELS:-401408}"
export OUTPUT="${OUTPUT:-results/trust_gate_1_offset20_${SLURM_JOB_ID:-local}.json}"

bash run_openeqa_ltu_smoke.slurm

python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUTPUT"])
if not out.is_file():
    print(f"WARN: missing {out}")
    raise SystemExit(0)

data = json.loads(out.read_text(encoding="utf-8"))
rows = data.get("ours") or []
for i, row in enumerate(rows):
    gold = (row.get("gold_answer") or "").strip()
    pred = (row.get("prediction") or "").strip()
    mem = ((row.get("debug") or {}).get("memorize") or {})
    print(f"pred={pred!r}")
    print(f"gold={gold!r}")
    print(
        f"direct_inserted={mem.get('direct_episodic_inserted')} "
        f"episodic_total={mem.get('episodic_total')} "
        f"errors={mem.get('direct_episodic_errors')}"
    )
    recent = mem.get("episodic_recent") or []
    if recent:
        details = (recent[0].get("details") or "")[:120]
        print(f"recent[0] details: {details!r}")
PY
