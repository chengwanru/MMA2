# OpenEQA 真实计时汇总（从已有 run 重算）

**生成方式:** `summarize_fair_speedup_from_existing.py`  
**原则:** 不用 report 里的 Direct `elapsed_sec`（只含 `generate`）；改用臂级墙钟 / 题数。

\[
\text{e2e 秒/题} = \frac{\texttt{direct\_wall\_sec}\ \text{或}\ \texttt{mma\_eval\_wall\_sec}\ \text{或}\ \texttt{direct\_sd\_wall\_sec}}{n}
\]

接受率（如有）为 token 加权：

\[
\text{acceptance} = \frac{\sum \text{draft\_tokens\_accepted}}{\sum \text{draft\_tokens\_proposed}}
\]

---

## 1. 各组真实时间（主表）

| 组 | frames | resize | e2e 秒/题 | 计时来源 | 接受率 | decode 秒/题 | decode/QA |
|----|--------|--------|-----------|----------|--------|--------------|-----------|
| Direct 8B (64f, no resize) | 64 | N | **55.603** | `direct_wall_sec/n` | n/a | — | — |
| Direct 8B (64f, resize) | 64 | Y | **4.099** | `direct_wall_sec/n` | n/a | — | — |
| Direct 8B (16f, no resize) | 16 | N | **0.641** | `direct_wall_sec/n` | n/a | — | — |
| MMA+SD (16f, no resize, no hybrid) | 16 | N | **36.013** | `mma_eval_wall_sec/n` | **90.4%** | 1.926 | 0.06 |
| MMA+SD baseline (64f, no resize, no hybrid) | 64 | N | **39.386** | `mma_eval_wall_sec/n` | **91.9%** | 1.834 | 0.05 |
| MMA+SD + visual routing (64f) | 64 | N | **39.598** | `mma_eval_wall_sec/n` | **91.8%** | 1.322 | 0.04 |
| MMA+SD (64f, resize) | 64 | Y | **87.181** | `mma_eval_wall_sec/n` | **92.0%** | 1.824 | 0.05 |
| MMA+SD + visual routing + block_verify (64f) | 64 | N | **33.018** | `mma_eval_wall_sec/n` | **92.5%** | 1.356 | 0.04 |
| MMA+SD + hybrid_vl (64f) | 64 | N | **44.171** | `mma_eval_wall_sec/n` | **93.5%** | 1.056 | 0.03 |
| MMA+SD + attach-3-frames (64f) | 64 | N | **40.438** | `mma_eval_wall_sec/n` | **88.3%** | 5.702 | 0.15 |
| MMA+SD + visual routing (64f, resize) | 64 | Y | **40.467** | `mma_eval_wall_sec/n` | **91.8%** | 1.339 | 0.04 |
| MMA+SD + visual routing (16f) | 16 | N | **39.785** | `mma_eval_wall_sec/n` | **88.7%** | 1.398 | 0.04 |
| MMA+SD + visual routing + attach-3-frames (64f) | 64 | N | **43.368** | `mma_eval_wall_sec/n` | **86.2%** | 5.241 | 0.13 |
| Pure SD, no MMA (64f, resize) | 64 | Y | **4.921** | `direct_sd_wall_sec/n` | **76.1%** | — | — |
| Pure SD, no MMA (16f, no resize) | 16 | N | **0.508** | `mean(elapsed_sec)` ⚠️ | **78.1%** | — | — |

⚠️ Pure SD@16 没有 `direct_sd_wall_sec`，仍是 generate-only 近似，**弱于**其它 REAL_WALL 行。

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

### 64f + no-resize（主设定）

| 方法 | Direct | 方法 | speedup |
|------|--------|------|---------|
| MMA+SD baseline | 55.603 | 39.386 | **1.412×** |
| MMA+SD + VR | 55.603 | 39.598 | **1.404×** |
| MMA+SD + VR + block_verify | 55.603 | 33.018 | **1.684×** |
| MMA+SD + hybrid_vl | 55.603 | 44.171 | **1.259×** |
| MMA+SD + attach-3 | 55.603 | 40.438 | **1.375×** |
| MMA+SD + VR + attach-3 | 55.603 | 43.368 | **1.282×** |

### 64f + resize

| 方法 | Direct | 方法 | speedup |
|------|--------|------|---------|
| MMA+SD plain | 4.099 | 87.181 | **0.047×** |
| MMA+SD + VR | 4.099 | 40.467 | **0.101×** |
| Pure SD | 4.099 | 4.921 | **0.833×** |

### 16f + no-resize

| 方法 | Direct | 方法 | speedup |
|------|--------|------|---------|
| MMA+SD | 0.641 | 36.013 | **0.018×** |
| MMA+SD + VR | 0.641 | 39.785 | **0.016×** |
| Pure SD | 0.641 | 0.508 | **1.262×** ⚠️（Pure SD 时间为 elapsed 近似） |

---

## 5. 读数注意

1. **主叙事用 §3.1 / §4 的 64f no-resize**：Direct ~55.6 s vs MMA ~33–44 s。  
2. **resize / 16f** 下 Direct 已经亚秒～数秒，MMA 仍 ~35–40 s，加速比 <1 是墙钟事实，不是接受率造假。  
3. MMA 接受率普遍 **~86–94%**，但 decode 只占 QA 的约 **3–6%**（attach-3 约 13–15%），所以高接受率不等于端到端大加速。  
4. `MMA+SD (64f, resize)` 的 **87 s** 明显高于其它 MMA（多半 memorize 被拉高），同设定内对比时单独看待。

---

## 6. 复现命令

```bash
cd /tmp/MMA2/MMA/public_evaluations/open_eqa
source use_mma_env.sh
$PY summarize_fair_speedup_from_existing.py --md results/fair_speedup_from_existing.md
```

本文件为人工整理的可读版，数字与上述脚本在 AIBox 上的一次运行输出一致。
