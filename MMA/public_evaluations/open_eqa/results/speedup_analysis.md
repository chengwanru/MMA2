# Speedup 分析（15组，4个对照组）

数据来源：本文件里的数字全部来自已经跑过的15组结果（`score_all_groups_report.py` 的 `GROUPS` 列表）和已有的debug/timing数据，**没有重新跑任何实验**。凡是现有数据里没有的字段，一律标 `N/A`，不做估算/凑数。

对照组固定为这4个：
1. Direct 8B（64帧，resize）
2. Direct 8B（64帧，no resize）
3. Direct 8B（16帧，no resize）
4. Pure SD，无MMA（64帧，resize）

加速比统一定义为 `speedup = 对照组时间 / 该组时间`（>1 表示该组比对照组快，<1 表示更慢）。

---

## 第一部分：15组原始时间数据

这里列出每组能拿到的所有时间口径：`e2e秒/题`是shell层面打点的真实墙钟（含模型加载/tokenize/检索/memorize/生成，一切都算在内）；`generate专用秒/题`只包含 `client.request()` 里 `generate()` 这一步本身（不含tokenize，但包含图像视觉编码，因为那发生在 `generate()` 内部）；`decode专用秒/题`只有MMA+SD组才有，是 draft+target+verify+rope 四步加总（不含tokenize、不含模型加载、不含memorize、不含检索）。

| # | 组 | frames | resize | e2e秒/题(墙钟) | 计时来源 | generate专用秒/题 | decode专用秒/题(仅MMA) | 接受率 |
|---|---|---|---|---|---|---|---|---|
| 1 | Direct 8B (64f, no resize) | 64 | N | 55.603 | `direct_wall_sec/n` | 46.146 | — | n/a |
| 2 | Direct 8B (64f, resize) | 64 | Y | 4.099 | `direct_wall_sec/n` | 0.873 | — | n/a |
| 3 | Direct 8B (16f, no resize) | 16 | N | 0.641 | `direct_wall_sec/n` | 0.298 | — | n/a |
| 4 | MMA+SD (16f, no resize, no hybrid) | 16 | N | 36.013 | `mma_eval_wall_sec/n` | N/A | 1.926 | 90.4% |
| 5 | MMA+SD baseline (64f, no resize, no hybrid) | 64 | N | 39.386 | `mma_eval_wall_sec/n` | N/A | 1.834 | 91.9% |
| 6 | MMA+SD + visual routing (64f) | 64 | N | 39.598 | `mma_eval_wall_sec/n` | N/A | 1.322 | 91.8% |
| 7 | MMA+SD (64f, resize) | 64 | Y | 87.181 | `mma_eval_wall_sec/n` | N/A | 1.824 | 92.0% |
| 8 | MMA+SD + visual routing + block_verify (64f) | 64 | N | 33.018 | `mma_eval_wall_sec/n` | N/A | 1.356 | 92.5% |
| 9 | MMA+SD + hybrid_vl (64f) | 64 | N | 44.171 | `mma_eval_wall_sec/n` | N/A | 1.056 | 93.5% |
| 10 | MMA+SD + attach-3-frames (64f) | 64 | N | 40.438 | `mma_eval_wall_sec/n` | N/A | 5.702 | 88.3% |
| 11 | MMA+SD + visual routing (64f, resize) | 64 | Y | 40.467 | `mma_eval_wall_sec/n` | N/A | 1.339 | 91.8% |
| 12 | MMA+SD + visual routing (16f) | 16 | N | 39.785 | `mma_eval_wall_sec/n` | N/A | 1.398 | 88.7% |
| 13 | MMA+SD + visual routing + attach-3-frames (64f) | 64 | N | 43.368 | `mma_eval_wall_sec/n` | N/A | 5.241 | 86.2% |
| 14 | Pure SD, no MMA (64f, resize) | 64 | Y | 4.921 | `direct_sd_wall_sec/n` | N/A（现有数据只有含加载墙钟，无单独generate值） | — | 76.1% |
| 15 | Pure SD, no MMA (16f, no resize) | 16 | N | 0.508 ⚠️ | `mean(elapsed_sec)`（近似，非真实墙钟，见下方说明） | 0.508 | — | 78.1% |

⚠️ 第15行的0.508，本质就是`generate专用秒/题`（因为它是`mean(elapsed_sec)`），只是当时没有走计时shell、没有真实的`direct_sd_wall_sec`，所以`e2e秒/题`这一列借用了这个近似值，比其它行的口径弱一档，第二/四部分里用到它做"端到端"对照时会再次标注。

`MMA+SD (64f, resize)` 的87.181秒明显高于同档其它MMA变体（此前分析怀疑是memorize被拉高），用于计算前建议单独核实。

---

## 第二部分：端到端（全部时间）加速比

两边都用"从这道题交给系统，到拿到最终答案文本"的完整墙钟时间，包含模型加载、tokenize、检索、memorize、生成在内的一切开销，谁都不排除；4个对照组各自的端到端时间就是它们自己的`e2e秒/题`（第一部分表里的数值）。

| # | 组 | vs Direct64resize(4.099s) | vs Direct64noresize(55.603s) | vs Direct16noresize(0.641s) | vs PureSD64resize(4.921s) |
|---|---|---|---|---|---|
| 1 | Direct 8B (64f, no resize) | 0.074× | 1.000× | 0.012× | 0.089× |
| 2 | Direct 8B (64f, resize) | 1.000× | 13.57× | 0.156× | 1.201× |
| 3 | Direct 8B (16f, no resize) | 6.40× | 86.74× | 1.000× | 7.68× |
| 4 | MMA+SD (16f, no hybrid) | 0.114× | 1.544× | 0.018× | 0.137× |
| 5 | MMA+SD baseline (64f) | 0.104× | 1.412× | 0.016× | 0.125× |
| 6 | MMA+SD + VR (64f) | 0.104× | 1.404× | 0.016× | 0.124× |
| 7 | MMA+SD (64f, resize) | 0.047× | 0.638× | 0.007× | 0.056× |
| 8 | MMA+SD + VR + block_verify (64f) | 0.124× | 1.684× | 0.019× | 0.149× |
| 9 | MMA+SD + hybrid_vl (64f) | 0.093× | 1.259× | 0.015× | 0.111× |
| 10 | MMA+SD + attach-3 (64f) | 0.101× | 1.375× | 0.016× | 0.122× |
| 11 | MMA+SD + VR (64f, resize) | 0.101× | 1.374× | 0.016× | 0.122× |
| 12 | MMA+SD + VR (16f) | 0.103× | 1.398× | 0.016× | 0.124× |
| 13 | MMA+SD + VR + attach-3 (64f) | 0.095× | 1.282× | 0.015× | 0.113× |
| 14 | Pure SD (64f, resize) | 0.833× | 11.30× | 0.130× | 1.000× |
| 15 | Pure SD (16f, no resize) ⚠️ | 8.07× | 109.5× | 1.262× | 9.69× |

⚠️ 第15行的对照数字用的是近似值（见第一部分说明），不是真实墙钟，仅供参考。第1-3行是Direct-8B跟自己（同算法、不同帧数/resize设置）比，反映的是"帧数/resize本身"对Direct的影响，跟MMA、跟SD无关，不能当"方法效果"引用。

---

## 第三部分：非端到端 —— MMA只算decode，其他只算generation（不含tokenize）

MMA+SD一侧只计算 draft+target+verify+rope 四步加总（`decode专用秒/题`），不含模型加载、检索、memorize、tokenize；Direct-8B 和 Pure SD 一侧只计算 `generate()` 这一步本身（同样不含tokenize，但包含图像视觉编码，因为这发生在`generate()`内部）。4个对照组里，`Pure SD (64f, resize)` 现有数据缺这个口径的独立数值，该列整列标 `N/A`。

| # | 组 | vs Direct64resize(0.873s) | vs Direct64noresize(46.146s) | vs Direct16noresize(0.298s) | vs PureSD64resize(N/A) |
|---|---|---|---|---|---|
| 1 | Direct 8B (64f, no resize) | 0.019× | 1.000× | 0.006× | N/A |
| 2 | Direct 8B (64f, resize) | 1.000× | 52.86× | 0.341× | N/A |
| 3 | Direct 8B (16f, no resize) | 2.93× | 154.85× | 1.000× | N/A |
| 4 | MMA+SD (16f, no hybrid) | 0.453× | 23.96× | 0.155× | N/A |
| 5 | MMA+SD baseline (64f) | 0.476× | 25.16× | 0.163× | N/A |
| 6 | MMA+SD + VR (64f) | 0.660× | 34.91× | 0.225× | N/A |
| 7 | MMA+SD (64f, resize) | 0.479× | 25.30× | 0.163× | N/A |
| 8 | MMA+SD + VR + block_verify (64f) | 0.644× | 34.03× | 0.220× | N/A |
| 9 | MMA+SD + hybrid_vl (64f) | 0.827× | 43.70× | 0.282× | N/A |
| 10 | MMA+SD + attach-3 (64f) | 0.153× | 8.09× | 0.052× | N/A |
| 11 | MMA+SD + VR (64f, resize) | 0.652× | 34.46× | 0.223× | N/A |
| 12 | MMA+SD + VR (16f) | 0.625× | 33.01× | 0.213× | N/A |
| 13 | MMA+SD + VR + attach-3 (64f) | 0.167× | 8.80× | 0.057× | N/A |
| 14 | Pure SD (64f, resize) | N/A（自身数据也缺） | N/A | N/A | N/A |
| 15 | Pure SD (16f, no resize) | 1.719× | 90.84× | 0.587× | N/A |

这一部分的大数字（几十倍）主要反映的是"Direct-8B处理真实图片要跑视觉编码器、MMA+SD用文字memory不用跑视觉编码器"这个差异，不是SD算法（draft先猜、target再核对）本身的加速——之前的分析已经确认过这一点，这里只是把算法列出来，不代表这组数字可以直接当"SD加速比"写进论文。

---

## 第四部分：MMA只算decode，其他算全部时间（含tokenize）

MMA+SD一侧仍然只计算 draft+target+verify+rope 四步加总（`decode专用秒/题`，跟第三部分一样）；Direct-8B 和 Pure SD 一侧改用完整的端到端墙钟时间（含模型加载、tokenize、检索、生成一切开销），即第二部分里同样的对照组数值。

| # | 组 | vs Direct64resize(4.099s) | vs Direct64noresize(55.603s) | vs Direct16noresize(0.641s) | vs PureSD64resize(4.921s) |
|---|---|---|---|---|---|
| 1 | Direct 8B (64f, no resize) | 0.074× | 1.000× | 0.012× | 0.089× |
| 2 | Direct 8B (64f, resize) | 1.000× | 13.57× | 0.156× | 1.201× |
| 3 | Direct 8B (16f, no resize) | 6.40× | 86.74× | 1.000× | 7.68× |
| 4 | MMA+SD (16f, no hybrid) | 2.13× | 28.87× | 0.333× | 2.56× |
| 5 | MMA+SD baseline (64f) | 2.24× | 30.32× | 0.350× | 2.68× |
| 6 | MMA+SD + VR (64f) | 3.10× | 42.06× | 0.485× | 3.72× |
| 7 | MMA+SD (64f, resize) | 2.25× | 30.48× | 0.351× | 2.70× |
| 8 | MMA+SD + VR + block_verify (64f) | 3.02× | 41.01× | 0.473× | 3.63× |
| 9 | MMA+SD + hybrid_vl (64f) | 3.88× | 52.66× | 0.607× | 4.66× |
| 10 | MMA+SD + attach-3 (64f) | 0.719× | 9.75× | 0.112× | 0.863× |
| 11 | MMA+SD + VR (64f, resize) | 3.06× | 41.52× | 0.479× | 3.68× |
| 12 | MMA+SD + VR (16f) | 2.93× | 39.77× | 0.459× | 3.52× |
| 13 | MMA+SD + VR + attach-3 (64f) | 0.782× | 10.61× | 0.122× | 0.939× |
| 14 | Pure SD (64f, resize) | 0.833× | 11.30× | 0.130× | 1.000× |
| 15 | Pure SD (16f, no resize) ⚠️ | 8.07× | 109.5× | 1.262× | 9.69× |

⚠️ 第15行"其他算全部时间"这一边，实际用的还是`mean(elapsed_sec)`近似值（这组从来没有测过真实的端到端墙钟），比同表其它行的口径弱一档。第1-3行同样是Direct跟自己比，仅供参照，不是方法效果。

---

## 各部分对照速查

| 部分 | MMA+SD 一侧 | Direct-8B / Pure SD 一侧 | 回答的问题 |
|---|---|---|---|
| 二 | 全部时间（含加载/tokenize/检索/memorize/生成） | 全部时间（同左） | 实际跑一次，完整耗时谁快——最接近真实使用体验，但两边"大头"不一样（MMA是逐题重载模型的税，Direct是处理真图的开销），数字会因这两个巧合的大头而互相抵消或放大 |
| 三 | 仅decode（draft+target+verify+rope） | 仅generate（不含tokenize，含视觉编码） | 抽掉两边的"固定成本"之后纯算法/纯计算量的差异——但这个差异主要来自"要不要处理真实图片"，不是SD机制本身 |
| 四 | 仅decode（同三） | 全部时间（同二） | 一个不对称的口径：MMA扣掉了自己的固定成本，Direct/PureSD没扣——这一部分刻意保留是为了对比"如果MMA能把架构开销降到接近0，理论上限能到多少"，不能当作当前系统的真实加速比 |
