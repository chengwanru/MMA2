# prompt_tokens 六组实验结果 + VR(visual routing) 对token无效果的根因

**目的：** 用 `prompt_tokens`（tokenize后的完整输入长度，含文字+图像token）量化 MMA 用文字memory替代真图、以及visual routing(VR)筛选memory item，各自省了多少输入token。

**跑法：** `run_prompt_tokens_6groups.sh`，6组各30题（同一批题，64f和16f各自复用同一份INPUT，保证3组内部可比），GPU1，`2026-08-02` 跑的。

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

Memory routing计数（VR组）：
- mma64_vr: mean_before_routing=9.39, mean_after_routing=1.29, budget=512.0 (n=28)
- mma16_vr: mean_before_routing=7.55, mean_after_routing=1.31, budget=512.0 (n=29)

派生的token节省比例：
- 64f: Direct → MMA(无VR)：130748.1 → 2326.0，**省98.2%**
- 64f: MMA(无VR) → MMA(+VR)：2326.0 → 2326.0，**省0.0%**
- 16f: Direct → MMA(无VR)：1116.1 → 1927.0，**MMA反而多用72.7%**
- 16f: MMA(无VR) → MMA(+VR)：1927.0 → 1927.0，**省0.0%**

---

## 二、三个需要注意的异常，不能直接当"证明"引用

### 1. 64f下MMA省token是真实、扎实的证据

130,748 vs 2,326，98.2%的节省第一次用真实数据量化出来了，可以直接用。130,748这个数字本身也解释了为什么Direct-8B在64f无resize下要跑46秒——上十万token的视觉输入编码本来就贵。

### 2. VR对token数**零效果**——已排查到确定性根因，不是巧合也不是bug巧合触发

VR筛选memory item是真实发生的（9.39条→1.29条），但筛的是一份跟QA prompt文字**完全脱钩的独立副本**，对`prompt_tokens`必然没有影响。具体链路（带文件行号）：

**真正决定prompt大小的路径**：
- `get_qa_ranked_events()`（[openeqa_memory.py:6586](../openeqa_memory.py)）选出相关episodic事件
- `format_episodic_block_for_qa(selected)`（[openeqa_memory.py:2330-2358](../openeqa_memory.py)）把事件渲染成文字block（每条最多900字符细节，`OPENEQA_QA_EPISODIC_DETAILS_CHARS`控制）
- 这段文字直接拼进 `user_prompt`（[run_openeqa_one_sample.py:675](../run_openeqa_one_sample.py)）
- tokenize后算出 `prompt_len`（[speculative_memory_client.py:1110](../../../MMA/llm_api/speculative_memory_client.py)），这一步在真正开始生成**之前**就已完成

**VR实际操作的路径（独立、脱钩）**：
- `events_to_memory_items(selected)`（[openeqa_memory.py:2307-2327](../openeqa_memory.py)）从同一批`selected`事件转出**另一份独立拷贝**`memory_items`，走单独的参数传递（`retrieved_memories`），从未被拼回`chat`/prompt文字（唯一例外是`local_rag`基线分支，不在MMA+SD路径里）
- `route_text_memory_items`/VR筛选发生在`generate_with_speculative_memory`内部（[decoding.py:429-453](../../../MMA/speculative_memory/decoding.py)），**这时候prompt早就tokenize完了**
- 筛选后的`memory_items`只被转成一个KV-cache偏置（`memory_kv`），用于speculative decoding验证阶段的辅助信号，从未被拼回文字prompt
- `stats_out["memories_before_routing"]`/`["memories_after_routing"]`（[decoding.py:452-453](../../../MMA/speculative_memory/decoding.py)）记录的就是这份独立副本的筛选计数

**结论**：VR筛选"9条→1.3条"是真实且测量正确的，但架构上这份被筛选的`memory_items`和真正进prompt的episodic caption文字，是从同一批`selected`事件各自独立转出的两份拷贝，彼此没有连接。要让VR真正省token，需要让`route_text_memory_items`的筛选结果（或等价的预算控制）也去限制喂给`format_episodic_block_for_qa`的`selected`事件列表（[run_openeqa_one_sample.py:673-675](../run_openeqa_one_sample.py)），而不是像现在这样只影响一份仅用于KV偏置的独立副本。这是一处代码改动，目前还没有做。

### 2.1 目前的写法结论

**"VR省内存"这个说法，现有代码和数据都不支持**，不建议直接写进论文；如果要证明VR省token，需要先做上述代码改动，再重新测。

### 3. 16f下Direct的token数不成比例地小，需要先核实图片分辨率是否一致

64f Direct=130,748，16f Direct=1,116——帧数只降了4倍，token数却降了117倍，不成正比（正常按比例应该在32,687左右）。可能是64f和16f两次跑分用的图片本身分辨率/frame_cache构建方式不一致（不是同一批图片只是抽帧数量不同），需要先核实，否则"16f: MMA反而多用72.7%"这条结论的可靠性存疑。

---

## 三、后续待办

1. 核实64f/16f两组Direct用的图片分辨率是否一致（检查两次frame_cache构建参数）
2. 如果要拿VR的token节省当论文证据，需要先改代码让VR的筛选结果真正影响`format_episodic_block_for_qa`喂的事件列表，再重新测
3. 目前唯一可以直接引用的，是"64f: Direct→MMA(无VR)省98.2%"这一条
