# P0实验：target-only + 相同compact memory prompt + 不用SD

**目的：** 回应i1cc W1、Pa58 W1——隔离"用文字memory替代真图"和"SD算法本身"这两个变量各自的贡献。

**环境说明：** 2026-09-18晚A800集群被重装（模型、数据、代码全部清空），本次实验在重新搭建的环境上跑（模型放`/nix`本地NVMe，数据/结果放`/workspace`bosfs对象存储）。由于是全新环境，本次跑出来的绝对时间**不能直接跟论文里旧环境的39.386s/39.598s对比**，因此额外补跑了MBSD-30/MMA2.0-30在同一新环境下的对照。

**踩过的坑（跑之前修过）：**
1. `compute_draft_policy`每题都会无条件重新计算并覆盖`OPENEQA_MAX_DRAFT_STEPS`，不检查`MMA_SPECULATIVE_BASELINE`/`MMA_TARGET_ONLY`——导致第一次跑的时候部分题目仍然跑了真实的draft+verify（比如某题`draft_accepted=2/2`）。修复：加`OPENEQA_FORCE_MAX_DRAFT_STEPS=0`强制覆盖，重新跑，确认所有题`draft_tokens_proposed=0`。
2. 模型/数据挂在bosfs对象存储上，每题模型重载耗时约39秒（第一次跑30题实测），把模型迁移到`/nix`本地NVMe后，memorize阶段耗时从702.7秒降到约95-100秒。

## 同环境、同30题、三者对比（2026-09-19）

用同一份557题输入文件的前30题（`open-eqa-multimodal-p0_f64_557.json`），三个run除了"要不要用SD"，其它环境/代码/prompt完全一致：

| 组 | RUN_DIR | total_sec(mean) | qa_wall_sec(mean) | qa_decode_sec(mean) | 接受率 | prompt_tokens(mean) |
|---|---|---|---|---|---|---|
| Baseline（target-only，`OPENEQA_FORCE_MAX_DRAFT_STEPS=0`）| `/workspace/wanru/open_eqa_runs/p0_target_only_baseline_557_20260919_004517`（前30/557）| 72.885 | 62.979 | 1.077 | n/a（0 draft，确认无SD）| 2081.6 |
| MBSD（`MMA_ENABLE_VISUAL_ROUTING=0`）| `/workspace/wanru/open_eqa_runs/env_check_mbsd_30_20260919_004642` | 76.068 | 66.419 | 1.623 | 87.8%（weighted，28/30题有数据）| 2081.6 |
| MMA2.0（`MMA_ENABLE_VISUAL_ROUTING=1`）| `/workspace/wanru/open_eqa_runs/env_check_mma2_30_20260919_004644` | 72.588 | 62.992 | 1.28 | 87.8%（同上）| 2081.6 |

三组`prompt_tokens`均值完全一致（2081.6），确认三者用的确实是同一份compact memory prompt，唯一变量是有没有用draft+verify。

## 观察

- 三者`total_sec`都在72.6-76.1秒区间，**MBSD比baseline（不用SD）还慢**，MMA2.0跟baseline基本持平（72.588 vs 72.885，差距在噪声范围内）
- `qa_decode_sec`（纯draft+target+verify+rope）本身很小（1.0-1.6秒），只占`total_sec`的1.5%-2%左右，即使SD在decode这一层确实存在一些差异，也不足以在总时间上体现出来
- 样本量仅30题，需要557题跑完之后的完整数据确认这个方向是否稳定

## 待办

- 等557题baseline跑完，取相同的timing/acceptance统计口径，与论文原始MBSD(39.386s)/MMA2.0(39.598s)行一起放进最终对照表
- LLM-Match打分（需要API key，等用户可用时补做）
