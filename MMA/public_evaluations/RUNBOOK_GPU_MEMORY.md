# GPU 显存与可复现实验（LTU / Gadi 通用思路）

无法对「永远不会 OOM」做数学保证；下列流程用于 **把显存争用导致的假失败压到最低**，并在出问题时有据可查。

## 1. 根因类型（区分对待）

| 现象 | 含义 |
|------|------|
| 日志里 **`CUDA out of memory`**，且 **`response":"{}"`** → **`executable_plan`** 缺失 | 推理侧 OOM，指标会像 planner 全失败 |
| 同卡出现 **其它 PID 占大量显存**（`nvidia-smi`） | **多作业/多进程争用**，应先解决独占或换节点 |
| 无 OOM，但 **`invalid_reason.jsonl`** 里 Thor 报错多 | 才是 **环境交互 / 策略** 问题 |

## 2. 跑长实验前必做（推荐顺序）

1. **独占 GPU**：Slurm `--gres=gpu:1` 且分区策略允许多作业独占；Gadi 按队列申请整卡资源。  
2. **作业开始后立刻**（或在 smoke 脚本里已打印）**`nvidia-smi`**：确认无陌生大进程占满显存。  
3. **环境变量**（可选但推荐）：  
   `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`  
4. **LTU 上先跑 1 集 smoke**：  
   `sbatch run_embench_memory_smoke.sh`  
   结束后：  
   `grep -c "CUDA out of memory" .../embench_memcheck_<jobid>.log`  
   **期望为 0**。  
5. **再跑** regression / A/B/C；若长作业中途 OOM，以当日 **`nvidia-smi` / 日志** 为准判断是否争用。

## 3. 仍 OOM 时（工程侧）

- 降 **分辨率**、缩短 **上下文**、开启 **flash-attn / 量化**（按你们 MMA 栈）。  
- 换 **更大显存 GPU** 或 **更长 walltime 排队换节点**。

## 4. 汇报时可说的边界

- 可以说：「我们在 **独占 GPU + smoke 无 OOM** 条件下复现了正常 planning，与此前 **日志中存在 OOM 与同卡他进程** 的那次运行相对照。」  
- 避免说：「以后绝对不会 OOM」——除非运维层面承诺单卡独占且负载可控。

---

集群专用命令分拆见 [CLUSTER_LTU.md](CLUSTER_LTU.md) 与 [CLUSTER_NCI_GADI.md](CLUSTER_NCI_GADI.md)。
