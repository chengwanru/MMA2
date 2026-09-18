# Rebuttal 必要实验清单（对应Submission2219三份审稿意见）

## P0 — 决定性缺失baseline：target-only + 相同compact episodic prompt，不用SD

**对应质疑：**
- Reviewer i1cc, Weakness 1: "The decisive baseline is missing: the target model alone on the same compact prompt, with ordinary decoding... Without it, the 1.404× headline cannot be credited to the new components. Almost all of the speedup comes from episodic-text conditioning, which is the contribution of prior work (MMA). The claimed contribution and the measured benefit do not match."
- Reviewer Pa58, Weakness 1: "A critical baseline is missing: target-only decoding with the same episodic prompt but without speculative decoding... it is unclear whether the reported 1.404x speedup mainly comes from replacing visual frames with textual memory. A memory-only, no-SD baseline is needed to isolate the incremental value of MBSD."

**为什么必要：** 论文Table 1只有Qwen3-VL-8B（真图）、Pure SD（OOM）、MBSD、MMA2.0四行，没有"target用MBSD/MMA2.0同款2,326-token compact prompt、贪心解码、不用draft、不用SD"这一行。论文自己在4.4节写"decoding accounts for less than 5% of total online inference time"（Table 1 Decode列：1.834s/1.322s，相对39.386s/39.598s总时间），两位审稿人据此直接推断：如果加上这一行baseline，它的wall-clock大概率会落在MBSD/MMA2.0附近，因为决定总时间的不是SD机制，而是"用文字memory替代真图"这个更早的架构选择（MMA前作贡献）。目前1.404×这个headline claim没法把功劳分清楚。

**怎么跑：** `run_openeqa_eval.py --variants baseline`，用MBSD/MMA2.0同一份`--baseline_config`（现成代码路径，不需要新写代码），至少覆盖Table 1主设定（64帧原分辨率），理想情况下Table 2两个辅助设定也补上。记录LLM-Match和online wall-clock latency，直接加进Table 1/2作为第五行。

---

## P0 — exact-match acceptance 和 semantic-rescue acceptance 分开报

**对应质疑：**
- Reviewer i1cc, Weakness 5: "The semantic rescue accepts tokens the target would not emit, so the 91.8% acceptance rate is not comparable to lossless speculative decoding, and the 96.6% preservation figure is measured against MBSD rather than against the better-performing target."
- Reviewer Pa58, Weakness 2: "The verification rule is not standard lossless speculative decoding... the reported 91.8% verification-policy acceptance in the main setting is not directly comparable to standard exact-match acceptance. The paper should report exact-match acceptance and semantic-rescue acceptance separately and show how each affects answer quality."

**为什么必要：** 论文3.4节公式(8)已经写明acceptance = exact match OR embedding-similarity rescue，Table 1/2报的91.8%/91.9%等数字是两者合并后的结果，读者没法判断"真正贪心命中"占多少、"语义救回硬凑"占多少。两位审稿人都要求拆开报，且要说明语义救回对答案质量的影响。

**怎么跑：** 复用已有的`accept_reason_counts`/`semantic_rescue_samples`机制（[verify.py](MMA/MMA/speculative_memory/verify.py)、[decoding.py](MMA/MMA/speculative_memory/decoding.py)、[inspect_semantic_rescue.py](MMA/public_evaluations/open_eqa/inspect_semantic_rescue.py)），在跑P0第一项实验时顺手加`OPENEQA_DEBUG=1`，同一批跑分即可拿到这个拆分数据，不需要单独跑。

---

## P1 — 精确说明evidence-adaptive policy π(q,M)的具体阈值规则

**对应质疑：**
- Reviewer i1cc, Comment 1: "The adaptive policy π(q,M) is described only qualitatively, and there is no appendix, no prompt templates, no sanitization rules, and no code statement... I suggest the authors specifying π(q,M) precisely (features, thresholds, ranges of K and λ) and including the prompts."
- Reviewer Pa58, Weakness 3: "The evidence-adaptive policy that chooses the draft length K and bias strength is under-specified. The paper states that the policy uses evidence decisiveness, query-record alignment, question type, and conflict detection, but the exact rules and sensitivity are not fully described."
- Reviewer yyM8: "the adaptive policy is described only in terms of broad signals... not defined in enough detail to reproduce."

**为什么必要：** 论文3.3节只有定性描述("increases speculative drafting when the retrieved evidence is sufficiently decisive...")，没有具体阈值/公式，三份审稿意见都独立提到这是可复现性的硬伤（Reproducibility评分普遍不高）。

**怎么跑：** 不需要跑实验。已经从`compute_draft_policy`（[openeqa_memory.py:6426-6530](MMA/public_evaluations/open_eqa/openeqa_memory.py:6426)）里提取出完整的margin/alignment/question-type/conflict四个信号定义和K/λ阈值判断表（见本次会话前面消息），直接整理成appendix。

---

## P1 — 定义Eq.2的clamp记号，说明embedding来自哪一层

**对应质疑：**
- Reviewer i1cc, Comment 2: "Define the clamp notation in Eq. 2, and state which layer the embeddings come from."

**为什么必要：** 纯写作清晰度问题，公式$c_i=[0.45s_i+0.40u_i+0.15v_i]_0^1$里的$[\cdot]_0^1$是clamp到[0,1]，论文没解释；Eq.3的$\mathrm{Emb}_T$具体是target模型哪一层的embedding也没写明。

**怎么跑：** 不需要跑实验，纯文字补充。

---

## P2 — 报告memorize/offline memory构建成本（不排除在latency之外）

**对应质疑：**
- Reviewer i1cc, Weakness 4: "the cost of building the memory offline is excluded."
- Reviewer Pa58, Weakness 5: "the reported online latency excludes upstream episodic-memory construction."

**为什么必要：** 论文Limitations已经**主动披露**了这一点（"the reported online wall-clock latency excludes upstream episodic-memory construction"），但两位审稿人仍然把它列为weakness——说明披露不够，读者想要一个具体数字，而不只是一句免责声明。

**怎么跑：** 不需要新实验。用已有debug文件里的`memorize_sec`字段，按episode摊销（每个episode的memorize总耗时 / 该episode实际问的题数），整理成一张补充表格，附在正文或appendix。

---

## P2 — 补充更完整的accuracy-latency frontier

**对应质疑：**
- Reviewer i1cc, Weakness 3: "No accuracy-latency frontier is explored beyond three points."

**为什么必要：** Figure 3的Pareto图只有3个点（Direct/MBSD/MMA2.0）。

**怎么跑：** 不需要新实验。已有15组消融数据（含visual routing、block_verify、hybrid_vl、attach-3-frames等10种MBSD/MMA2.0变体），整理成补充图即可。

---

## 时间上大概率来不及，建议列为future work并在rebuttal中说明

- **人工验证/judge敏感性分析**（Reviewer yyM8: "Quality is evaluated using a single LLM judge, without human validation or judge-sensitivity analysis."）
- **多模态SD baseline对比**（Reviewer i1cc, Weakness 6: "No multimodal speculative-decoding baselines are run (DREAM, ViSpec, and others are only cited)"）
- **换benchmark/模型对**（Reviewer yyM8 和 Pa58 都提到single-benchmark、single-model-pair的局限）
