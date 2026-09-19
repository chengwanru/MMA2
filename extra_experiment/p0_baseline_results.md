# Extra Experiment: Target-only Baseline（不用SD，同一份compact memory prompt）

**实验设置：** 这一组让target模型（8B）单独用跟MBSD/MMA2.0完全相同的那份压缩后的episodic memory prompt直接生成答案，**不使用draft模型，不做speculative decoding**——跟MBSD/MMA2.0唯一的区别就是"要不要用SD"，其它（memory构建、prompt内容、模型、环境）全部一致。

**样本量：** 557题。

---

## 结果

正确性确认：557题全部无报错，全部确认`draft_tokens_proposed=0`（真正的target-only，没有被SD污染）。

| 指标 | 均值 | 中位数 |
|---|---|---|
| total_sec（含memorize） | 119.659 | 58.271 |
| memorize_sec | 60.423 | 0.0 |
| qa_wall_sec（不含memorize） | 59.236 | 57.952 |
| qa_decode_sec | 1.17 | 1.118 |
| prompt_tokens | 2257.8 | — |

说明：`total_sec`均值被少数需要新建episode记忆的题目拉高（大部分题memorize_sec=0，复用已有记忆），中位数更能反映典型单题成本；`qa_wall_sec`不含memorize，是跨组比较时更合适的口径。
