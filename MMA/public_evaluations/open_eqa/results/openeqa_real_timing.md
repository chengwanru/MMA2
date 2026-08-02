# OpenEQA 真实计时汇总（从已有 run 重算）

**生成方式:** `summarize_fair_speedup_from_existing.py`  
**原则:** 不用 report 里的 Direct `elapsed_sec`（只含 `generate`）；改用臂级墙钟 / 题数。

> **论文可用性总览（先读这个再看下面的表）：**
> 下面所有数字都是真实测量（不是编造/hardcode），但只有 **§3.1 / §4「64f + no-resize」这一档** 目前达到可以直接写进论文的可信度。
> **64f resize 和 16f 这两档的 <1x「MMA 更慢」结论，现阶段不要当成「SD 算法本身不起作用」的证据去写**——原因见 §5a。



\[
\text{e2e 秒/题} = \frac{\texttt{direct\_wall\_sec}\ \text{或}\ \texttt{mma\_eval\_wall\_sec}\ \text{或}\ \texttt{direct\_sd\_wall\_sec}}{n}
\]

接受率（如有）为 token 加权：

\[
\text{acceptance} = \frac{\sum \text{draft\_tokens\_accepted}}{\sum \text{draft\_tokens\_proposed}}
\]

---

## 1. 各组真实时间（主表）

| 组 | frames | resize | e2e 秒/题 | 计时来源 | 接受率 | decode 秒/题 | decode/QA | 论文可用性 |
|----|--------|--------|-----------|----------|--------|--------------|-----------|-----------|
| Direct 8B (64f, no resize) | 64 | N | **55.603** | `direct_wall_sec/n` | n/a | — | — | ✅ 可用 |
| Direct 8B (64f, resize) | 64 | Y | **4.099** | `direct_wall_sec/n` | n/a | — | — | ⚠️ 数值真实，但仅作为 resize 档分母，见 §5a |
| Direct 8B (16f, no resize) | 16 | N | **0.641** | `direct_wall_sec/n` | n/a | — | — | ⚠️ 数值真实，但仅作为 16f 档分母，见 §5a |
| MMA+SD (16f, no resize, no hybrid) | 16 | N | **36.013** | `mma_eval_wall_sec/n` | **90.4%** | 1.926 | 0.06 | ❌ 见 §5a（重载税未扣除） |
| MMA+SD baseline (64f, no resize, no hybrid) | 64 | N | **39.386** | `mma_eval_wall_sec/n` | **91.9%** | 1.834 | 0.05 | ✅ 可用 |
| MMA+SD + visual routing (64f) | 64 | N | **39.598** | `mma_eval_wall_sec/n` | **91.8%** | 1.322 | 0.04 | ✅ 可用 |
| MMA+SD (64f, resize) | 64 | Y | **87.181** | `mma_eval_wall_sec/n` | **92.0%** | 1.824 | 0.05 | ⚠️ 异常值，用前需查明原因，见 §5b |
| MMA+SD + visual routing + block_verify (64f) | 64 | N | **33.018** | `mma_eval_wall_sec/n` | **92.5%** | 1.356 | 0.04 | ✅ 可用 |
| MMA+SD + hybrid_vl (64f) | 64 | N | **44.171** | `mma_eval_wall_sec/n` | **93.5%** | 1.056 | 0.03 | ✅ 可用 |
| MMA+SD + attach-3-frames (64f) | 64 | N | **40.438** | `mma_eval_wall_sec/n` | **88.3%** | 5.702 | 0.15 | ✅ 可用 |
| MMA+SD + visual routing (64f, resize) | 64 | Y | **40.467** | `mma_eval_wall_sec/n` | **91.8%** | 1.339 | 0.04 | ❌ 见 §5a（重载税未扣除） |
| MMA+SD + visual routing (16f) | 16 | N | **39.785** | `mma_eval_wall_sec/n` | **88.7%** | 1.398 | 0.04 | ❌ 见 §5a（重载税未扣除） |
| MMA+SD + visual routing + attach-3-frames (64f) | 64 | N | **43.368** | `mma_eval_wall_sec/n` | **86.2%** | 5.241 | 0.13 | ✅ 可用 |
| Pure SD, no MMA (64f, resize) | 64 | Y | **4.921** | `direct_sd_wall_sec/n` | **76.1%** | — | — | ❌ 见 §5a（重载税未扣除） |
| Pure SD, no MMA (16f, no resize) | 16 | N | **0.508** | `mean(elapsed_sec)` ⚠️ | **78.1%** | — | — | ❌ 双重问题：见 §5a + 下方 ⚠️ |

⚠️ Pure SD@16 没有 `direct_sd_wall_sec`（当时启动方式没走计时shell），仍是 generate-only 近似，**弱于**其它 REAL_WALL 行，口径比同表其它行更弱一档。

---

## 2. 旧 report 的 `elapsed` vs 真实墙钟（Direct）

说明旧 ablation report 为何会把 Direct 估得过快：

| 组 | 旧 `elapsed` 均值 | 真实 e2e（wall/n） | wall / sum(elapsed) |
|----|-------------------|--------------------|---------------------|
| Direct@64 no-resize | 46.146 s | **55.603 s** | ≈ **1.20×** |
| Direct@64 resize | 0.873 s | **4.099 s** | ≈ **4.70×** |
| Direct@16 no-resize | 0.298 s | **0.641 s** | ≈ **2.15×** |

---

## 3. 按设定分组看时间（便于对照，先不除）

### 3.1 frames=64, resize=N

| 方法 | e2e 秒/题 | 接受率 |
|------|-----------|--------|
| Direct 8B | **55.603** | — |
| MMA+SD baseline | **39.386** | 91.9% |
| MMA+SD + VR | **39.598** | 91.8% |
| MMA+SD + VR + block_verify | **33.018** | 92.5% |
| MMA+SD + hybrid_vl | **44.171** | 93.5% |
| MMA+SD + attach-3 | **40.438** | 88.3% |
| MMA+SD + VR + attach-3 | **43.368** | 86.2% |

### 3.2 frames=64, resize=Y

| 方法 | e2e 秒/题 | 接受率 |
|------|-----------|--------|
| Direct 8B | **4.099** | — |
| Pure SD | **4.921** | 76.1% |
| MMA+SD plain | **87.181** | 92.0% |
| MMA+SD + VR | **40.467** | 91.8% |

### 3.3 frames=16, resize=N

| 方法 | e2e 秒/题 | 接受率 |
|------|-----------|--------|
| Direct 8B | **0.641** | — |
| Pure SD | **0.508** ⚠️ | 78.1% |
| MMA+SD | **36.013** | 90.4% |
| MMA+SD + VR | **39.785** | 88.7% |

---

## 4. 同设定加速比（附录；仅同 frames+resize 相除）

定义: \(\text{speedup} = \text{Direct e2e} / \text{方法 e2e}\)（>1 表示方法比 Direct 快）。

### 64f + no-resize（主设定，✅ 唯一可直接写论文的档位）

| 方法 | Direct | 方法 | speedup |
|------|--------|------|---------|
| MMA+SD baseline | 55.603 | 39.386 | **1.412×** |
| MMA+SD + VR | 55.603 | 39.598 | **1.404×** |
| MMA+SD + VR + block_verify | 55.603 | 33.018 | **1.684×** |
| MMA+SD + hybrid_vl | 55.603 | 44.171 | **1.259×** |
| MMA+SD + attach-3 | 55.603 | 40.438 | **1.375×** |
| MMA+SD + VR + attach-3 | 55.603 | 43.368 | **1.282×** |

### 64f + resize（❌ 见 §5a，重载税未扣除，不要当算法结论引用）

| 方法 | Direct | 方法 | speedup |
|------|--------|------|---------|
| MMA+SD plain | 4.099 | 87.181 | 0.047×（且分子异常值，见 §5b.1） |
| MMA+SD + VR | 4.099 | 40.467 | 0.101× |
| Pure SD | 4.099 | 4.921 | 0.833× |

### 16f + no-resize（❌ 见 §5a，重载税未扣除，不要当算法结论引用）

| 方法 | Direct | 方法 | speedup |
|------|--------|------|---------|
| MMA+SD | 0.641 | 36.013 | 0.018× |
| MMA+SD + VR | 0.641 | 39.785 | 0.016× |
| Pure SD | 0.641 | 0.508 | 1.262× ⚠️（Pure SD 时间为 elapsed 近似，双重问题见 §5a+主表备注） |

---

## 5. 读数注意

1. **主叙事用 §3.1 / §4 的 64f no-resize**：Direct ~55.6 s vs MMA ~33–44 s，**加速比 1.26×–1.68×，这是目前唯一可以直接写论文的档位**。
2. MMA 接受率普遍 **~86–94%**，但 decode 只占 QA 的约 **3–6%**（attach-3 约 13–15%），所以高接受率不等于端到端大加速——这个结论本身独立于下面 §5a 的问题，是可信的。

### 5a. resize / 16f 档「MMA 更慢」为什么现在不能直接写论文

这两档下 Direct 单题只要零点几秒到几秒，而 MMA+SD 走的是逐题独立子进程（`run_openeqa_eval.py` 每题 `subprocess.run` 一次 `run_openeqa_one_sample.py`），**每一题都要把 2B 草稿模型 + 8B 目标模型从磁盘重新加载一次**；Direct/Pure SD 那边是全部 557 题跑在同一个进程里（`LLMClient.create()` 有全局缓存，模型只加载一次，摊到 557 题里几乎不占时间）。

也就是说 `mma_eval_wall_sec` 里混了两种性质完全不同的时间：
- SD 算法本身的推理耗时（我们真正关心的）
- 纯工程实现选择（逐题重启进程）导致的模型重载税（和 SD 算法好不好无关，Direct 完全不用付这个税）

这笔税在 Direct 本身很慢的 64f no-resize 档被摊薄看不出来（所以那一档结果可信）；但在 resize / 16f 档，Direct 本身已经快到亚秒级，这笔固定税就成了压倒性的大头，把任何 SD 带来的真实收益完全淹没。**所以 resize / 16f 档测出的 <1x，大概率主要反映的是"重启两个模型要多久"，而不是"SD 算法在小图/少帧场景下不起作用"。**

在这一点被扣除之前，**不要**把这两档写成"SD 在小规模输入下没有优势"这类算法层面的结论；要写的话应明确标注为"当前逐题重启进程的工程开销尚未扣除，结果待重测"。

修正路径：这次已经往 `speculative_memory_client.py` / `run_openeqa_one_sample.py` 里加了 `model_load_sec`（每题模型重载单独计时，见 commit `a707dbf`/`6e67e30`），下次重跑后可以算：

```
mma_algorithmic_wall_sec = mma_eval_wall_sec − Σ(model_load_sec across all samples)
```

拿这个扣税后的数字再和 Direct 比，才是真正只对比算法本身、和工程实现无关的公平加速比。

### 5b. 其它需要在引用前确认的点

1. `MMA+SD (64f, resize)` 的 **87 s** 明显高于其它 MMA（多半 memorize 被拉高），同设定内对比时需单独查明原因，不要直接拿来算加速比。
2. `Direct 8B (64f, no resize)` / `(64f, resize)` 两组的原始结果文件存在 `/workspace/open_eqa_runs/...`（该盘之前确认过存在写入不稳定的情况，读取通常没问题）；引用这两组数字前建议再确认一下没有受那次已知问题影响。
3. Pure SD@16f 除了 §5a 的重载税问题，自身计时口径也比其它行弱一档（见上方 ⚠️ 备注），不建议在同一张表里和 REAL_WALL 行直接相除比较。

---

## 6. 复现命令

```bash
cd /tmp/MMA2/MMA/public_evaluations/open_eqa
source use_mma_env.sh
$PY summarize_fair_speedup_from_existing.py --md results/fair_speedup_from_existing.md
```

本文件为人工整理的可读版，数字与上述脚本在 AIBox 上的一次运行输出一致。
