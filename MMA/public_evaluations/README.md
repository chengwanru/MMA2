# MMA public_evaluations

Benchmark drivers are grouped by dataset. Shared agent/metrics code lives under `common/`.

## Layout

| Directory | Benchmark | Entry points |
|-----------|-----------|--------------|
| `common/` | Shared | `agent.py`, `evals.py`, `metrics/` |
| `locomo/` | LOCOMO / ScreenshotVQA | `run_instance.py`, `main.py`, `run.sh` |
| `fever/` | FEVER | `run_fever_eval.py` |
| `embodiedbench/` | EmbodiedBench (eb-alf) | `embodiedbench_server.py`, `run_embench_mma_one_node*.sh`, `submit_*.pbs` |
| `open_eqa/` | Open-EQA (multimodal) | `run_openeqa_eval.py`, `make_openeqa_multimodal.py` |

## Quick start

```bash
# LOCOMO (single conversation index)
cd MMA/public_evaluations/locomo
python run_instance.py --agent_name mma --dataset LOCOMO --global_idx 0

# FEVER
cd MMA/public_evaluations/fever
python run_fever_eval.py --fever_data_path data/paper_dev.jsonl --limit 100

# EmbodiedBench + MMA server (cluster scripts vary; see embodiedbench/docs/)
cd MMA/public_evaluations/embodiedbench
python check_embodiedbench_deps.py
bash run_embench_memory_smoke_gadi.sh   # Gadi example

# Open-EQA
cd MMA/public_evaluations/open_eqa
python run_openeqa_eval.py --input_file data/open-eqa-multimodal.json --output_file results/out.json
```

## Data paths

- LOCOMO: `locomo/data/locomo10.json`
- FEVER: `fever/data/paper_dev.jsonl`
- Open-EQA: `open_eqa/data/` (see `make_openeqa_multimodal.py`)
- EmbodiedBench scenes/assets: external `EmbodiedBench` clone + optional `open_eqa/data/scene_datasets/` (gitignored when large)

## PYTHONPATH

Scripts add `MMA/public_evaluations` to `sys.path` so `from common.agent import AgentWrapper` works. The `mma` package is loaded from `MMA/` (parent of `public_evaluations`).
