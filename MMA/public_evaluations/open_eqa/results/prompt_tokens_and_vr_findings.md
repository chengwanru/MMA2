# prompt_tokens 六组实验结果 + VR(visual routing) 路径追踪

**跑法：** `run_prompt_tokens_6groups.sh`，6组各30题（同一批题，64f和16f各自复用同一份INPUT），GPU1，2026-08-02。

---

## 一、六组结果

| 组 | RUN_DIR | n | mean_tokens | min | max |
|---|---|---|---|---|---|
| direct64 | `/nix/openeqa_work/open_eqa_runs/token_test_direct64_20260802_234409` | 30 | 130748.1 | 130744 | 130756 |
| mma64_novr | `/nix/openeqa_work/open_eqa_runs/token_test_mma64_novr_20260802_234409` | 30 | 2326.0 | 727 | 2888 |
| mma64_vr | `/nix/openeqa_work/open_eqa_runs/token_test_mma64_vr_20260802_234409` | 30 | 2326.0 | 727 | 2888 |
| direct16 | `/nix/openeqa_work/open_eqa_runs/token_test_direct16_20260802_234409` | 30 | 1116.1 | 1112 | 1124 |
| mma16_novr | `/nix/openeqa_work/open_eqa_runs/token_test_mma16_novr_20260802_234409` | 30 | 1927.0 | 403 | 2860 |
| mma16_vr | `/nix/openeqa_work/open_eqa_runs/token_test_mma16_vr_20260802_234409` | 30 | 1927.0 | 403 | 2860 |

Memory routing计数（VR组，来自`stats_out["memories_before_routing"]`/`["memories_after_routing"]`）：
- mma64_vr: mean_before=9.39, mean_after=1.29, budget=512.0 (n=28)
- mma16_vr: mean_before=7.55, mean_after=1.31, budget=512.0 (n=29)

派生比例：
- 64f: Direct → MMA(无VR)：130748.1 → 2326.0（省98.2%）
- 64f: MMA(无VR) → MMA(+VR)：2326.0 → 2326.0（省0.0%）
- 16f: Direct → MMA(无VR)：1116.1 → 1927.0（MMA多用72.7%）
- 16f: MMA(无VR) → MMA(+VR)：1927.0 → 1927.0（省0.0%）

---

## 二、VR对prompt_tokens无效果——路径追踪（文件行号）

**决定prompt文字大小的路径：**
- `get_qa_ranked_events()`（[openeqa_memory.py:6586](../openeqa_memory.py)）→ `format_episodic_block_for_qa(selected)`（[openeqa_memory.py:2330-2358](../openeqa_memory.py)，渲染成文字block）→ 拼进`user_prompt`（[run_openeqa_one_sample.py:675](../run_openeqa_one_sample.py)）→ tokenize得到`prompt_len`（[speculative_memory_client.py:1110](../../../MMA/llm_api/speculative_memory_client.py)），此步在生成开始前完成

**VR实际操作的路径：**
- `events_to_memory_items(selected)`（[openeqa_memory.py:2307-2327](../openeqa_memory.py)）从同一批`selected`事件转出独立拷贝`memory_items`，经`retrieved_memories`单独传递，从未拼回`chat`/prompt文字（MMA+SD路径下）
- `route_text_memory_items`筛选发生在`generate_with_speculative_memory`内部（[decoding.py:429-453](../../../MMA/speculative_memory/decoding.py)），此时prompt已tokenize完毕
- 筛选后的`memory_items`只转成KV-cache偏置（`memory_kv`），用于speculative decoding验证阶段，不影响`prompt_ids`

## 三、64f/16f的Direct token数不成比例

64f Direct=130,748，16f Direct=1,116，帧数降4倍但token数降了117倍（按比例预期约32,687）。
