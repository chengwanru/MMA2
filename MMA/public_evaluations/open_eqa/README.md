# OpenEQA（LTU）

在LTU + `mma_speculative_memory.yaml` 跑 **baseline / ours**，输出 `results/*.json`。

## 流程

1. `make_openeqa_multimodal.py` — 生成 `data/open-eqa-multimodal.json`
2. `run_openeqa_eval.py` — 同一 yaml 跑两遍（baseline 自动关 memory/draft）

## 环境

LTU 用 group miniconda 的 **`embench`** 环境（与 EmbodiedBench 相同），脚本会自动 activate。

```bash
export MMA_OFFLINE=1   # required: memory agents use local Qwen (not gpt-4o-mini API)
export MMA_MEMORY_SEARCH_METHOD=bm25   # required on HPC: no llama_index / OpenAI embeddings
export MMA_SPECULATIVE_OFFLOAD_TARGET=1   # 40GB A100: offload part of 8B target to CPU for QA memory KV
export OPENEQA_ABSORB_BATCH_SIZE=4   # 16 frames -> 4 absorbs
export OPENEQA_SPLIT_PHASES=1   # memorize + QA in separate processes (avoids QA OOM after absorb)
export OPENEQA_DEBUG=1   # episodic + BM25 debug in results/openeqa_debug/*.json and slurm log
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

## 0. 检查数据（LTU 上跑）

OpenEQA 共 **1636** 题、**152** 个 episode：**hm3d 63** + **scannet 89**。两者缺一都会 skip 题目。

```bash
cd /data/group/zhaolab/project/MMA2/MMA/public_evaluations/open_eqa
python check_openeqa_data.py
# 期望: hm3d 63/63, scannet 89/89, missing 0
```

若只有 hm3d、没有 scannet：Mac 上 `bash upload_from_mac.sh --with-data`（会下 hm3d + scannet 两个包）。

## 1. 多模态 JSON

默认 **全帧**（`--all_frames`）；快速 smoke 可只取 16 帧：

```bash
python make_openeqa_multimodal.py --all_frames --max_samples 5
# 或: --frames_per_episode 16 --frame_sampling uniform
```

## 2. 跑评测

**Memorize（写入 episodic）**：`OPENEQA_SPLIT_PHASES=1` 时 memorize 子进程使用 **Qwen3-VL-8B baseline + tool call**（`MMA_BASELINE_TOOLS=1`，仅 episodic agent，`MMA_TARGET_ONLY=1` 省显存）。若 tool call 未写入 DB，自动 **fallback** 到 direct episodic insert（VL caption + `insert_event`）。memorize 结束要求 `episodic_memory` 行数 > 0。

**QA**：独立子进程，**ours** 路径用 draft+target speculative memory + BM25 检索（不设 `MMA_BASELINE_TOOLS`）。

相关 env（smoke 脚本已默认）：

| 变量 | memorize | QA |
|------|----------|-----|
| `OPENEQA_EPISODIC_TOOL_CALL=1` | 8B baseline tool absorb | — |
| `OPENEQA_EPISODIC_ONLY=1` | 只跑 episodic agent | — |
| `MMA_TARGET_ONLY=1` | 只加载 8B | off |
| `OPENEQA_DIRECT_EPISODIC=1` | tool 失败时 fallback | — |
| `OPENEQA_QA_BASELINE=0` | — | ours speculative memory |

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

**推荐：8 帧 tool-call smoke（先验证 episodic 写入）：**

```bash
cd /data/group/zhaolab/project/MMA2
git pull
cd MMA/public_evaluations/open_eqa
mkdir -p logs
sbatch run_openeqa_ltu_toolcall_smoke.slurm
tail -f logs/openeqa_toolcall_smoke_<jobid>.log
```

跑完后检查 log 里应有 `episodic_total>0`，或：

```bash
python3 - <<'PY'
import json
r = json.load(open("results/toolcall_smoke_<jobid>.json"))["ours"][0]
print("gold:", r["gold_answer"])
print("pred:", r["prediction"])
print("debug memorize episodic_total:", r.get("debug",{}).get("memorize",{}).get("episodic_total"))
PY
```

**一条全帧：**

```bash
cd /data/group/zhaolab/project/MMA2/MMA/public_evaluations/open_eqa
git pull
mkdir -p logs
sbatch run_openeqa_ltu_one_fullframe.slurm
tail -f logs/openeqa_one_fullframe_<jobid>.log
```

Wall time **24h**（LTU `day` 分区上限）。约 80–120 帧 / 条；若超时改用 `-p week -t 48:00:00`。

**快速 16 帧 smoke：**

```bash
ALL_FRAMES=0 FRAMES_PER_EPISODE=16 LIMIT=1 MAX_SAMPLES=1 sbatch run_openeqa_ltu_smoke.slurm
```

默认 smoke（`run_openeqa_ltu_smoke.slurm`）：全帧、`MAX_SAMPLES=10`、`LIMIT=2`。

```bash
mkdir -p logs
sbatch run_openeqa_ltu_smoke.slurm
```

默认 `--limit 5`。改条数：`LIMIT=20 sbatch run_openeqa_ltu_smoke.slurm`

看日志：

```bash
tail -f logs/openeqa_smoke_<jobid>.log
```

全量：`sbatch run_openeqa_ltu.slurm`

## 官方 LLM-Match 打分

### 用哪个 API？

| 阶段 | 推荐 | 原因 |
|------|------|------|
| **Smoke / 迭代（现在）** | **[OpenRouter 免费模型](https://openrouter.ai)** | OpenAI 兼容、注册即有 key、有 `*:free` 模型；10 条 smoke 够用 |
| **备选** | **[Groq](https://console.groq.com)** | 免费层、速度快；作 OpenRouter 限流时的 backup |
| **正式实验 / 论文** | **OpenAI GPT-4** | 与 OpenEQA 论文一致（`gpt-4-1106-preview`） |

Smoke 分数**不要**和论文表格直接比；只用于 ours vs baseline 的相对趋势。正式数只用 `--judge-profile official`。

### 配置流程（Smoke，OpenRouter 推荐）

**1. 注册并拿 key**

- 打开 https://openrouter.ai/keys → 创建 API key（免费层约 **50 次/天**，10 条 smoke 足够）
- 可选 backup：https://console.groq.com/keys

**2. 写本地 env（不要 commit）**

```bash
cd MMA/public_evaluations/open_eqa
cp openeqa_judge_smoke.env.example openeqa_judge_smoke.env
# 编辑 OPENROUTER_API_KEY=sk-or-v1-...
```

**3. 安装 judge 依赖（一次）**

```bash
python3 -m pip install "openai>=1.3" "numpy>=1.26" "tenacity>=8.2"
```

**4. 跑完 MMA 推理后打分**

```bash
set -a && source openeqa_judge_smoke.env && set +a

python run_openeqa_official_score.py \
  --input_file results/direct_episodic_bias_tuned_10_<jobid>.json \
  --variant ours \
  --judge-profile openrouter-free \
  --force \
  --dry-run
```

- `--force`：允许 partial（不足 1636 题）
- `--dry-run`：只评 5 题省 quota；去掉则评全部已导出题目
- 分数写入 `results/metrics/<name>-openrouter-free-metrics.json`

**5. 正式实验（论文数字）**

```bash
export OPENAI_API_KEY=sk-...
bash setup_openeqa_official_scorer.sh   # 仅需一次

python run_openeqa_official_score.py \
  --input_file results/openeqa_full.json \
  --variant ours \
  --judge-profile official
```

### Judge profiles

| `--judge-profile` | Key 环境变量 | 默认模型 |
|-------------------|-------------|----------|
| `openrouter-free` | `OPENROUTER_API_KEY` | `google/gemma-3-27b-it:free` |
| `groq-free` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| `official` | `OPENAI_API_KEY` | `gpt-4-1106-preview` |

换模型：`OPENEQA_OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free` 或 `--judge-model ...`

限流时在 env 里设 `OPENEQA_JUDGE_SLEEP_SEC=1.0`（两次请求间隔秒数）。

### 仅导出、不调用 API

```bash
python export_openeqa_official.py \
  --input_file results/direct_episodic_bias_tuned_10_<jobid>.json \
  --output_file results/official/predictions.json \
  --variant ours
```

## Baidu AIBox 共享环境（`/workspace` + `/tmp`）

Pod 重启后先激活共享 Python 3.11（包在 `/tmp/embench_staging`，备份在 `/workspace/conda_envs/site-packages_backup.tar`）：

```bash
source /workspace/MMA2/MMA/public_evaluations/open_eqa/use_mma_env.sh
export CUDA_VISIBLE_DEVICES=0   # 选显存最多的 GPU
cd /workspace/MMA2/MMA/public_evaluations/open_eqa
$PY run_openeqa_eval.py ...
```

## 常见报错

| 报错 | 处理 |
|------|------|
| `No module named mma` | `cd MMA2/MMA && cp -a MMA mma`（bosfs 勿用 symlink） |
| multimodal 条数为 0 | 确认 `../data/open_eqa_data/hm3d-v0/*.tar` 存在 |
| inode 满 / 磁盘占满 | 勿整包解压 tar；跑 `bash cleanup_openeqa_extracted.sh` |
| CUDA OOM（40G GPU） | 默认 smoke 只跑 `VARIANTS=ours`、`LIMIT=2`、2 帧；baseline 另开 job：`VARIANTS=baseline sbatch ...` |
