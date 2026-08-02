#!/usr/bin/env bash
# 6-group prompt_tokens experiment: Direct-8B vs MMA+SD(no VR) vs MMA+SD(+VR), at 64f and 16f.
# Goal: quantify how much input-token footprint MMA's text-memory (vs raw images) and
# visual routing (memory-item pruning) save, per question, on the SAME 30 questions.
#
# Usage (run on GPU 0, from open_eqa dir):
#   bash run_prompt_tokens_6groups.sh
#
# All 6 runs are sequential (same GPU), each on the same 30 questions per frame count.
# At the end, prints a summary table (mean/min/max prompt_tokens per group, plus
# memories_before/after_routing for the two VR groups) straight to stdout/log.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

STAMP="$(date +%Y%m%d_%H%M%S)"
WORK_ROOT=/nix/openeqa_work
export CUDA_VISIBLE_DEVICES=0
export FRAMES_ROOT=/workspace/openeqa_data
export MODEL_ROOT=/workspace/models/qwen3-vl
export MMA_DRAFT_MODEL_PATH=/workspace/models/qwen3-vl/Qwen3-VL-2B-Instruct
export MMA_TARGET_MODEL_PATH=/workspace/models/qwen3-vl/Qwen3-VL-8B-Instruct
export OPENEQA_CAPTION_CACHE=/nix/openeqa_work/openeqa_caption_cache_v6
export OPENEQA_COLLECT_SD_STATS=1
export OPENEQA_VERIFY_REJECT_BAD_DRAFT=0
export PRECOMPUTE_CAPTIONS=0
export OPENEQA_DEBUG=1
export WORK_ROOT

LIMIT=30
INPUT_64F=/workspace/open-eqa-multimodal-compare_o0_n557_f64_20260727_224853.json

declare -A RUN_DIRS

echo "########################################"
echo "# Group 1/6: Direct-8B, 64f"
echo "########################################"
RUN_NAME="token_test_direct64_${STAMP}"
INPUT="${INPUT_64F}" ARMS=direct OFFSET=0 LIMIT="${LIMIT}" FRAMES_PER_EPISODE=64 \
  RUN_NAME="${RUN_NAME}" bash run_openeqa_aibox_compare_direct_vs_mma.sh
RUN_DIRS[direct64]="${WORK_ROOT}/open_eqa_runs/${RUN_NAME}"

echo "########################################"
echo "# Group 2/6: MMA+SD baseline (no VR), 64f"
echo "########################################"
RUN_NAME="token_test_mma64_novr_${STAMP}"
unset MMA_ENABLE_VISUAL_ROUTING || true
INPUT="${INPUT_64F}" ARMS=mma OFFSET=0 LIMIT="${LIMIT}" FRAMES_PER_EPISODE=64 \
  OPENEQA_HOME_ROOT="${WORK_ROOT}/openeqa_home_${RUN_NAME}" \
  RUN_NAME="${RUN_NAME}" bash run_openeqa_aibox_compare_direct_vs_mma.sh
RUN_DIRS[mma64_novr]="${WORK_ROOT}/open_eqa_runs/${RUN_NAME}"

echo "########################################"
echo "# Group 3/6: MMA+SD + visual routing, 64f"
echo "########################################"
RUN_NAME="token_test_mma64_vr_${STAMP}"
export MMA_ENABLE_VISUAL_ROUTING=1
INPUT="${INPUT_64F}" ARMS=mma OFFSET=0 LIMIT="${LIMIT}" FRAMES_PER_EPISODE=64 \
  OPENEQA_HOME_ROOT="${WORK_ROOT}/openeqa_home_${RUN_NAME}" \
  RUN_NAME="${RUN_NAME}" bash run_openeqa_aibox_compare_direct_vs_mma.sh
unset MMA_ENABLE_VISUAL_ROUTING
RUN_DIRS[mma64_vr]="${WORK_ROOT}/open_eqa_runs/${RUN_NAME}"

echo "########################################"
echo "# Group 4/6: Direct-8B, 16f (building a fresh 16f/30-sample INPUT first)"
echo "########################################"
RUN_NAME="token_test_direct16_${STAMP}"
ARMS=direct OFFSET=0 LIMIT="${LIMIT}" FRAMES_PER_EPISODE=16 \
  RUN_NAME="${RUN_NAME}" bash run_openeqa_aibox_compare_direct_vs_mma.sh
RUN_DIRS[direct16]="${WORK_ROOT}/open_eqa_runs/${RUN_NAME}"
INPUT_16F="$(grep -oP '^INPUT=\K.*' "${RUN_DIRS[direct16]}/run.log" | head -1)"
echo ">>> Captured 16f INPUT for reuse: ${INPUT_16F}"

echo "########################################"
echo "# Group 5/6: MMA+SD baseline (no VR), 16f"
echo "########################################"
RUN_NAME="token_test_mma16_novr_${STAMP}"
INPUT="${INPUT_16F}" ARMS=mma OFFSET=0 LIMIT="${LIMIT}" FRAMES_PER_EPISODE=16 \
  OPENEQA_HOME_ROOT="${WORK_ROOT}/openeqa_home_${RUN_NAME}" \
  RUN_NAME="${RUN_NAME}" bash run_openeqa_aibox_compare_direct_vs_mma.sh
RUN_DIRS[mma16_novr]="${WORK_ROOT}/open_eqa_runs/${RUN_NAME}"

echo "########################################"
echo "# Group 6/6: MMA+SD + visual routing, 16f"
echo "########################################"
RUN_NAME="token_test_mma16_vr_${STAMP}"
export MMA_ENABLE_VISUAL_ROUTING=1
INPUT="${INPUT_16F}" ARMS=mma OFFSET=0 LIMIT="${LIMIT}" FRAMES_PER_EPISODE=16 \
  OPENEQA_HOME_ROOT="${WORK_ROOT}/openeqa_home_${RUN_NAME}" \
  RUN_NAME="${RUN_NAME}" bash run_openeqa_aibox_compare_direct_vs_mma.sh
unset MMA_ENABLE_VISUAL_ROUTING
RUN_DIRS[mma16_vr]="${WORK_ROOT}/open_eqa_runs/${RUN_NAME}"

echo ""
echo "########################################"
echo "# All 6 runs done. RUN_DIRS:"
for k in direct64 mma64_novr mma64_vr direct16 mma16_novr mma16_vr; do
  echo "  ${k}: ${RUN_DIRS[$k]}"
done
echo "########################################"
echo ""
echo "########################################"
echo "# SUMMARY (prompt_tokens + memory routing)"
echo "########################################"

python3 - "${RUN_DIRS[direct64]}" "${RUN_DIRS[mma64_novr]}" "${RUN_DIRS[mma64_vr]}" \
          "${RUN_DIRS[direct16]}" "${RUN_DIRS[mma16_novr]}" "${RUN_DIRS[mma16_vr]}" <<'PYEOF'
import sys
import json
from pathlib import Path
from score_all_groups_report import _load_rows, _row_debug
from inspect_acceptance_and_timing import prompt_tokens_stats, _spec_stats_for_row

dirs = sys.argv[1:7]
labels = ["direct64", "mma64_novr", "mma64_vr", "direct16", "mma16_novr", "mma16_vr"]
specs = [
    ("direct64", Path(dirs[0]) / "direct_vl8b.json.raw.json", "direct_sd"),
    ("mma64_novr", Path(dirs[1]) / "mma_sd.json", "ours"),
    ("mma64_vr", Path(dirs[2]) / "mma_sd.json", "ours"),
    ("direct16", Path(dirs[3]) / "direct_vl8b.json.raw.json", "direct_sd"),
    ("mma16_novr", Path(dirs[4]) / "mma_sd.json", "ours"),
    ("mma16_vr", Path(dirs[5]) / "mma_sd.json", "ours"),
]

print(f"{'group':14} {'n':>4} {'mean_tok':>10} {'min':>6} {'max':>6}")
print("-" * 46)
results = {}
for label, path, variant in specs:
    if not path.is_file():
        print(f"{label:14} FILE NOT FOUND: {path}")
        continue
    rows = _load_rows(path, variant)
    stats = prompt_tokens_stats(rows, variant)
    results[label] = stats
    if stats is None:
        print(f"{label:14} no prompt_tokens data found ({len(rows)} rows)")
        continue
    print(f"{label:14} {stats['n']:>4} {stats['mean']:>10} {stats['min']:>6} {stats['max']:>6}")

print()
print("Memory routing (VR groups): memories_before_routing / after_routing / budget")
for label in ("mma64_vr", "mma16_vr"):
    path = dict(zip(labels, dirs))[label]
    p = Path(path) / "mma_sd.json"
    if not p.is_file():
        continue
    rows = _load_rows(p, "ours")
    befores, afters, budgets = [], [], []
    for r in rows:
        spec = _spec_stats_for_row(r, "ours")
        if not spec:
            continue
        if spec.get("memories_before_routing") is not None:
            befores.append(spec["memories_before_routing"])
        if spec.get("memories_after_routing") is not None:
            afters.append(spec["memories_after_routing"])
        if spec.get("routing_budget") is not None:
            budgets.append(spec["routing_budget"])
    if befores:
        print(
            f"  {label}: mean_before={sum(befores)/len(befores):.2f} "
            f"mean_after={sum(afters)/len(afters):.2f} "
            f"mean_budget={sum(budgets)/len(budgets):.1f} (n={len(befores)})"
        )
    else:
        print(f"  {label}: no memories_before/after_routing data found")

print()
if results.get("direct64") and results.get("mma64_novr"):
    d, m = results["direct64"]["mean"], results["mma64_novr"]["mean"]
    print(f"64f: Direct->MMA(no VR) token reduction: {d} -> {m} ({100*(1-m/d):.1f}% smaller)")
if results.get("mma64_novr") and results.get("mma64_vr"):
    m0, m1 = results["mma64_novr"]["mean"], results["mma64_vr"]["mean"]
    print(f"64f: MMA(no VR)->MMA(+VR) token reduction: {m0} -> {m1} ({100*(1-m1/m0):.1f}% smaller)")
if results.get("direct16") and results.get("mma16_novr"):
    d, m = results["direct16"]["mean"], results["mma16_novr"]["mean"]
    print(f"16f: Direct->MMA(no VR) token reduction: {d} -> {m} ({100*(1-m/d):.1f}% smaller)")
if results.get("mma16_novr") and results.get("mma16_vr"):
    m0, m1 = results["mma16_novr"]["mean"], results["mma16_vr"]["mean"]
    print(f"16f: MMA(no VR)->MMA(+VR) token reduction: {m0} -> {m1} ({100*(1-m1/m0):.1f}% smaller)")
PYEOF
