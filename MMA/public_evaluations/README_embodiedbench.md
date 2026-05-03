# EmbodiedBench + MMA Server 常见错误与修复

**集群分拆（请勿混用命令）**：[**LTU（Slurm）**](CLUSTER_LTU.md) · [**NCI Gadi（PBS）**](CLUSTER_NCI_GADI.md) · [**GPU 显存 / 可复现核对**](RUNBOOK_GPU_MEMORY.md)

---

## 1. `No module named 'httpx_sse'`

**现象**：EmbodiedBench 客户端收到：
```text
Error: {"error":"No module named 'httpx_sse'","response":"{}"}
```

**原因**：运行 `embodiedbench_server.py` 的 Python 环境里没有安装 `httpx_sse`。服务端在处理请求时会间接用到该依赖（如 MIRIX/OpenAI SSE 流式接口），未安装则报错并返回上述 JSON。

**修复**：在**启动 server 的那台机器、同一个 conda/venv 环境**里安装（PyPI 包名是**连字符** `httpx-sse`，import 时用下划线 `httpx_sse`）：
```bash
pip install httpx-sse
```

然后重启 `embodiedbench_server.py`。

**若你已装过依赖仍报错**：多半是「装依赖的 Python」和「启动 server 的 Python」不是同一个。请在同一环境下先检查：
```bash
cd MMA/public_evaluations
python check_embodiedbench_deps.py
```
脚本会打印当前 `sys.executable` 并检测 `httpx_sse`、`flask`、`opentelemetry.instrumentation.requests` 等是否可 import。

---

## 2. `No module named 'opentelemetry.instrumentation.requests'`

**现象**：客户端收到：
```text
Error: {"error":"No module named 'opentelemetry.instrumentation.requests'","response":"{}"}
```

**原因**：server 端在加载 mma/mirix 时会 import OpenTelemetry 的 requests 插桩，环境里未安装该包。

**修复**：在**启动 server 的同一环境**里安装：
```bash
pip install opentelemetry-instrumentation-requests
```
然后重启 `embodiedbench_server.py`。

---

## 3. `cannot import name 'PreTrainedConfig' from 'transformers.configuration_utils'`

**现象**：客户端收到：
```text
Error: {"error":"cannot import name 'PreTrainedConfig' from 'transformers.configuration_utils' (...)","response":"{}"}
```

**原因**：server 端某依赖（或 mma 推理栈）会从 `transformers.configuration_utils` 里 import `PreTrainedConfig`，当前环境里的 **transformers** 版本过新、过旧或安装异常，导致该导入失败。

**修复**：在**启动 server 的同一环境**（如 `embench`）里重装/升级 transformers：

```bash
# 先试升级到最新稳定版
pip install --upgrade transformers
```

若升级后其它包报错，可改钉到一个兼容版本再试，例如：

```bash
pip install "transformers>=4.30,<4.46"
```

然后重启 `embodiedbench_server.py`。

---

## 4. `An unexpected error occurred: 'executable_plan'` 与 `Planner Output Action: -1`

**现象**：客户端报 `'executable_plan'`、`Planner Output Action: -1`，且每条都伴随服务端返回的 error（如 `httpx_sse`、`opentelemetry.instrumentation.requests` 等）。

**原因**：服务端因缺少依赖返回了 `{"error":"...", "response":"{}"}`，`response` 里没有合法 JSON，也没有 `executable_plan` 字段。EmbodiedBench 解析不到 `executable_plan` 就会报错并退化为 action -1。

**修复**：按上面修好对应缺失依赖（如 `httpx-sse`、`opentelemetry-instrumentation-requests`），保证服务端能正常返回带 `executable_plan` 的 JSON，这两个错误会一起消失。

---

## 5. 推荐：在 server 环境一次性装齐依赖

在运行 `embodiedbench_server.py` 的环境里建议安装 MMA 的依赖（包含 `httpx-sse`）：

```bash
cd /path/to/MMA2/MMA
pip install -r requirements.txt
# 若用 requirements-mma-env.txt（无 PyQt6/ffmpeg 等）：
# pip install -r requirements-mma-env.txt
```

然后再启动 server。

---

## 6. `summary.json` 里 `planner_output_error` 很高、`num_steps` 为 0

**现象**：EmbodiedBench 能跑完 episode，但 `planner_output_error` 接近 planner 调用次数，且 `num_steps` 为 0。

**常见原因**：自定义 MMA 服务端对 planner JSON 做了**过严校验**（例如用正则从 prompt 里抽 `allowed_action_ids`，抽得不全或与模型输出的 id 不一致），校验失败就返回 **HTTP 500 + `response: "{}"`**，客户端统计为 planner 错误且没有可执行步。

**仓库内默认行为（推荐）**：

- **默认关闭** action_id 白名单校验；EmbodiedBench 仍会在执行阶段校验动作是否合法。
- 在返回前会根据 prompt 里形如 **`数字: 描述`** 的动作表，对 **`action_name` 与某一行描述能对上的步** 自动**纠正 `action_id`**（减少模型胡填 id 的情况）。

**可选环境变量**：

| 变量 | 作用 |
|------|------|
| `EMBODIEDBENCH_ENFORCE_ACTION_ALLOWLIST=1` | 重新开启「action_id 必须在正则抽到的集合里」的严格校验（一般仅在调试对比时用）。 |
| `EMBODIEDBENCH_TRACE_LOG=/path/to/planner_trace.log` | 校验失败时把原因和截取后的 JSON 追加写入该文件，便于对照修复。 |

修改后需**重启** `embodiedbench_server.py`（或重提 Slurm 作业）。

---

## 7. 减少「选错物体 / 重复 find」类 invalid action（服务端规划提示）

`embodiedbench_server.py` 支持在转发给模型前前置英文规划规则（对齐 TASK、只用动作表里的 id、避免无关家具、少重复同一步）。

> 注意：该提示默认**关闭**（某些任务会回归）。仅建议做 A/B 对照时显式开启。

| 变量 | 作用 |
|------|------|
| `EMBODIEDBENCH_ENABLE_PLANNER_HINTS=1` | 显式开启规划提示（默认关闭）。 |
| `EMBODIEDBENCH_PLANNER_HINT_TEXT='...'` | 自定义整段提示（覆盖默认；勿写 `数字: 描述` 行，以免干扰动作表解析）。 |
| `EMBODIEDBENCH_DISABLE_LOOP_BREAKER=1` | 关闭“避免重复上一轮首步 action_id”的断环逻辑（用于 A/B 对照定位根因）。 |
| `EMBODIEDBENCH_DISABLE_FAILURE_FEEDBACK_HINT=1` | 关闭“把上一轮失败首步写入下一轮提示”的反馈机制（默认开启）。 |
| `EMBODIEDBENCH_ENABLE_FIRST_ACTION_GUARD=1` | 开启“首步硬约束”：首步若是与任务无关的物体（如 Safe/KeyChain）会尽量换成 `find` 任务物体或导航。`embodiedbench_server` 默认关闭；`run_embench_mma_one_node.sh` 默认导出为 `1`（可用 `EMBODIEDBENCH_ENABLE_FIRST_ACTION_GUARD=0` 关掉）。 |
| `EMBODIEDBENCH_SIM_INFO_LEVEL` | 仿真器信息回传分档：`off`（默认，关闭）/ `A`（只传上一轮 env feedback）/ `B`（A + 紧凑状态提示，如 visible/reachable/holding/collision）/ `C`（B + 原始上下文摘录）。建议按 A→B→C 逐步 A/B。 |
| `EMBODIEDBENCH_ACTION_CATALOG_OBJECT_HINT=1` | **可选**：从 prompt 里 **ACTION LIST** 中所有 `find a …` 目标解析出一行 **Find targets:**，插在 planner 输入前，收窄物体词汇（思路类似 RoboAgent 从技能表抽物体表）。默认关闭；建议与 `EMBODIEDBENCH_ENABLE_PLANNER_HINTS` 等小样本 A/B。 |

改完后需**重启** server 或重提 Slurm。

**找本次 run 的目录（不要用字面量 `...`，也不要假设 shell 里已有 `EXP_NAME`）**：

- `run_embench_memory_smoke.sh` 会把 `EXP_NAME=...` 打在 **`${EB_ROOT}/embench_memcheck_<jobid>.log`** 开头（与 `#SBATCH -o` 一致）。例如 job `402896`：

```bash
EB_ROOT="${EB_ROOT:-/data/group/zhaolab/project/EmbodiedBench}"
JOB=402896
# Do not use cut -d= -f2- on a line that also contains VAR=value — use sed to take EXP_NAME token only.
EXP_NAME=$(grep -m1 '^EXP_NAME=' "${EB_ROOT}/embench_memcheck_${JOB}.log" | sed -n 's/^EXP_NAME=\([^[:space:]]*\).*/\1/p')
BASE="${EB_ROOT}/running/eb_alfred/mma_${EXP_NAME}/base"
echo "EXP_NAME=${EXP_NAME}" "BASE=${BASE}"
test -f "${BASE}/invalid_reason.jsonl" && grep -c "put down the object in hand" "${BASE}/invalid_reason.jsonl" || echo "missing invalid_reason.jsonl"
test -f "${BASE}/planner_trace.log" && grep -c "put down the object in hand" "${BASE}/planner_trace.log" || echo "missing planner_trace.log (older smoke runs: resubmit smoke script or set EMBODIEDBENCH_TRACE_LOG before sbatch)"
```

- 若未提前记 `EXP_NAME`，也可用：`ls -td "${EB_ROOT}/running/eb_alfred/mma_memcheck_smoke_"*/base/results/summary.json 2>/dev/null | head -1` 找最近一次 memcheck smoke 的 `summary.json`，再 `dirname` 两次得到 `base/`。

---

## 8. Invalid action 专项：回归集、诊断、可行性门控、短闭环（实施清单）

### 8.1 上游 EmbodiedBench 补丁（invalid 原因 JSONL + 反馈字段）

在 **EmbodiedBench 仓库根目录**执行（或查看说明）：

- [`patches/embodiedbench_upstream/README.md`](patches/embodiedbench_upstream/README.md)
- `bash patches/embodiedbench_upstream/apply_patches.sh /path/to/EmbodiedBench`

作用：`EBAlfEnv` 在 Thor 交互失败时写入 `EMBODIEDBENCH_INVALID_LOG_JSONL`；`CustomModel` / `VLMPlanner` 将上一轮 `env_feedback` 以表单字段 `last_env_feedback` 发给 MMA server（与 [`embodiedbench_server.py`](embodiedbench_server.py) 对应）。

### 8.2 固定 20 集回归

- 列表文件：[`regression_episodes_base.json`](regression_episodes_base.json)（通过 **`+selected_indexes=[...]`** 传给 Hydra；必须带 **`+`**，否则报 `Key 'selected_indexes' is not in struct`）。
- 一键脚本：[`run_embench_regression.sh`](run_embench_regression.sh)（内部 `down_sample_ratio=1` 且传 `+selected_indexes=...`）。Slurm 会把脚本拷到 spool 目录执行，故用环境变量 **`ROOT` / `MMA_PEV`**（默认 `/data/group/zhaolab/project`）定位 `regression_episodes_base.json`，勿依赖 `sbatch` 时的当前目录。

### 8.3 结果汇总脚本

```bash
python MMA/public_evaluations/scripts/summarize_invalid_actions.py \
  /path/to/EmbodiedBench/running/eb_alfred/mma_<exp>/base/results \
  --invalid-log /path/to/invalid_reason.jsonl
```

### 8.4 MMA server 环境变量（可行性门控 / 短计划）

| 变量 | 作用 |
|------|------|
| `EMBODIEDBENCH_FEASIBILITY_GATE=1` | 一键预设：首步 guard + 短 `executable_plan`（默认最多 3 步，可用 `EMBODIEDBENCH_EXECUTABLE_PLAN_MAX_LEN` 覆盖）。 |
| `EMBODIEDBENCH_SHORT_HORIZON_PLAN=1` | 未显式设置 `EMBODIEDBENCH_EXECUTABLE_PLAN_MAX_LEN` 时，把计划截断为 **2** 步。 |
| `EMBODIEDBENCH_EXECUTABLE_PLAN_MAX_LEN` | 显式最大步数（覆盖短 horizon 默认值）。 |
| `EMBODIEDBENCH_ENABLE_FIRST_ACTION_GUARD` | 首步物体对齐 guard（`run_embench_mma_one_node.sh` 默认 `1`）。 |

客户端（EmbodiedBench 打补丁后）可传 `last_env_feedback`，server 会插入 `[Simulator feedback from previous step]`。

推荐对照设置（仅示例）：

```bash
# A 档：最保守，只把上一轮失败原因带回下一轮
export EMBODIEDBENCH_SIM_INFO_LEVEL=A

# B 档：A + 紧凑状态摘要（更利于短步 replan）
export EMBODIEDBENCH_SIM_INFO_LEVEL=B

# C 档：B + 更丰富上下文（工程上更强，但公平性风险更高）
export EMBODIEDBENCH_SIM_INFO_LEVEL=C
```

若 EmbodiedBench 客户端（或你自定义评测器）愿意传结构化字段，server 会优先使用这些字段（不传也兼容）：

- `sim_info_json`（JSON 字符串；可一次性承载全部键）
- 或单字段表单：`reason_code`, `last_action_id`, `last_action_name`, `holding_object`,
  `target_visible`, `target_reachable`, `visible_objects_topk`, `agent_pose_summary`,
  `step_idx`, `episode_progress`

建议最小可用集合（先做 B 档）：`reason_code + target_visible + target_reachable + holding_object`。

### 8.4.1 Sim-info 四档（并行）

- **小样本快扫**（`down_sample_ratio=0.01`，`eval_sets=[base]`）：[`run_embench_siminfo_quick.sh`](run_embench_siminfo_quick.sh)  
  `bash run_embench_siminfo_quick.sh` → 报告 `${EB_ROOT}/embench_siminfo_quick_<TS>.txt`（需已 `git pull` 含 B 行 job id 解析修复）。
- **固定 20 集 regression**（`DOWNSAMPLE=1`，`+selected_indexes` 来自 `regression_episodes_base.json`，四 job 并行）：[`run_embench_siminfo_regression.sh`](run_embench_siminfo_regression.sh)  
  `bash run_embench_siminfo_regression.sh` → 报告 `${EB_ROOT}/embench_siminfo_regression_<TS>.txt`。默认 **`PARTITION=week`**、**`-t 72:00:00`**（`day` 仅允许 24h，勿用默认直接投 `day`）。  
  **若在 `day` 上跑**：设 **`PARTITION=day TIME_LIMIT=24:00:00`**，且整 20 集常会 `TIMEOUT`，请**分两趟**（各约一半 index，`EXP_NAME` 带 `_c0` / `_c1`）：  
  `REGRESSION_CHUNK=0 bash run_embench_siminfo_regression.sh`  
  `REGRESSION_CHUNK=1 bash run_embench_siminfo_regression.sh`  
  指标需**合并两趟**的 `summary.json` / per-episode 结果后再对比。

### 8.5 A/B/C/D 消融矩阵

脚本：[`run_embench_ablation.sh`](run_embench_ablation.sh)（Slurm 提交四条 job）。  
**晋升标准**（与计划一致）：在同一 `selected_indexes` 上，`num_invalid_actions` 中位数较基线下降 ≥40%，且 `planner_output_error` 不明显变差。

### 8.6 晋升闸门（手动检查）

1. `python scripts/summarize_invalid_actions.py ...`  
2. 若 `reason_code` 直方图以 `not_visible` / `not_reachable` 为主，优先增加探索/导航再 pick，而不是继续加长 JSON 计划。
