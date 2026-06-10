# OpenEQA（LTU）

在LTU + `mma_speculative_memory.yaml` 跑 **baseline / ours**，输出 `results/*.json`。

## 流程

1. `make_openeqa_multimodal.py` — 生成 `data/open-eqa-multimodal.json`
2. `run_openeqa_eval.py` — 同一 yaml 跑两遍（baseline 自动关 memory/draft）

## 环境

LTU 用 group miniconda 的 **`embench`** 环境（与 EmbodiedBench 相同），脚本会自动 activate。

```bash
export MMA_OFFLINE=1   # required: routes memory agents to local Qwen (not gpt-4o-mini API)
export HF_HOME=/data/group/zhaolab/project/hf_cache
```

工作目录：

```bash
cd /data/group/zhaolab/project/MMA2/MMA/public_evaluations/open_eqa
```

## 存储（重要）

**长期保留：** `open-eqa-v0.json` + `hm3d-v0/*.tar` + `scannet-v0/*.tar`（HF 下载）

**不要**在共享盘上整包解压 tar（会爆 inode）。`make_openeqa_multimodal.py` 只会从每个 tar 里抽出评测需要的少量 PNG 到 `data/frame_cache/`。

Slurm smoke 默认把 cache 放在 `$SLURM_TMPDIR`（作业结束自动清理）。

若之前误解压过整包 episode 目录，清理：

```bash
bash cleanup_openeqa_extracted.sh
# 或指定 tar 所在目录：
bash cleanup_openeqa_extracted.sh /data/group/zhaolab/project/MMA2/MMA/public_evaluations/data/open_eqa_data
```

## 1. 多模态 JSON

```bash
cd /data/group/zhaolab/project/MMA2/MMA/public_evaluations/open_eqa
source /data/group/zhaolab/project/miniconda/bin/activate embench

python make_openeqa_multimodal.py --max_samples 20
# 只缓存少量 PNG；评测完可 rm -rf data/frame_cache
```

## 2. 跑评测

MMA 按 **Episodic Memory** 流程：每条样本先 `memorizing=True` + `force_absorb_content=True` 写入 episode 帧（本地 VL 默认要攒满 20 帧才 absorb，OpenEQA 帧数少必须强制），再 `memorizing=False` 只问问题（与 LOCOMO 相同）。

先试 2 条：

```bash
python run_openeqa_eval.py \
  --input_file data/open-eqa-multimodal.json \
  --output_file results/smoke.json \
  --baseline_config ../../configs/mma_speculative_memory.yaml \
  --ours_config ../../configs/mma_speculative_memory.yaml \
  --limit 20
```

全量：去掉 `--limit`。

## Slurm smoke（LTU）

```bash
cd /data/group/zhaolab/project/MMA2/MMA/public_evaluations/open_eqa
mkdir -p logs
sbatch run_openeqa_ltu_smoke.slurm
```

默认 `--limit 5`。改条数：`LIMIT=20 sbatch run_openeqa_ltu_smoke.slurm`

看日志：

```bash
tail -f logs/openeqa_smoke_<jobid>.log
```

全量：`sbatch run_openeqa_ltu.slurm`

## 常见报错

| 报错 | 处理 |
|------|------|
| `No module named mma` | `cd MMA2/MMA && ln -sfn MMA mma` |
| multimodal 条数为 0 | 确认 `../data/open_eqa_data/hm3d-v0/*.tar` 存在 |
| inode 满 / 磁盘占满 | 勿整包解压 tar；跑 `bash cleanup_openeqa_extracted.sh` |
| CUDA OOM（40G GPU） | 默认 smoke 只跑 `VARIANTS=ours`、`LIMIT=2`、2 帧；baseline 另开 job：`VARIANTS=baseline sbatch ...` |
