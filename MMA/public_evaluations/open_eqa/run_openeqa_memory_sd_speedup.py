#!/usr/bin/env python3
"""
OpenEQA 纯推测解码测速脚本 (支持 Ablation Study: Baseline vs Vanilla SD vs Routed SD)
"""

import argparse
import gc
import json
import os
from pathlib import Path
from contextlib import contextmanager

from openeqa_memory import (
    ensure_episodic_from_frames,
    list_reranked_episodic_for_question,
    select_events_for_qa,
    events_to_memory_items,
    format_episodic_block_for_qa,
    prepare_draft_policy_for_agent,
    wipe_mma_sqlite,
    is_yes_no_question
)
from mma.llm_api.llm_client import LLMClient
from mma.schemas.llm_config import LLMConfig
from run_openeqa_one_sample import _init_agent, _format_eqa_question

def _parse_args():
    parser = argparse.ArgumentParser(description="纯测带记忆的 OpenEQA SD 加速比 (支持路由消融实验)")
    parser.add_argument("--input_file", required=True, help="OpenEQA JSON 数据集")
    parser.add_argument("--output_file", required=True, help="结果输出")
    parser.add_argument("--config", required=True, help="MMA 配置文件")
    parser.add_argument("--limit", type=int, default=None, help="样本数限制")
    return parser.parse_args()

@contextmanager
def _ablation_env(mode: str):
    """
    控制底层开关，实现三种消融模式：
    - baseline: 纯 8B (无 SD，无 Routing)
    - vanilla_sd: 原生 SD (全量记忆，无 Routing)
    - routed_sd: 路由 SD (精简记忆 + SD)
    """
    keys = ("MMA_SPECULATIVE_BASELINE", "MMA_TARGET_ONLY", "MMA_ENABLE_VISUAL_ROUTING")
    saved = {k: os.environ.get(k) for k in keys}
    
    if mode == "baseline":
        os.environ["MMA_SPECULATIVE_BASELINE"] = "1"
        os.environ["MMA_TARGET_ONLY"] = "1"
        os.environ["MMA_ENABLE_VISUAL_ROUTING"] = "0"
    elif mode == "vanilla_sd":
        os.environ.pop("MMA_SPECULATIVE_BASELINE", None)
        os.environ.pop("MMA_TARGET_ONLY", None)
        os.environ["MMA_ENABLE_VISUAL_ROUTING"] = "0"
    elif mode == "routed_sd":
        os.environ.pop("MMA_SPECULATIVE_BASELINE", None)
        os.environ.pop("MMA_TARGET_ONLY", None)
        os.environ["MMA_ENABLE_VISUAL_ROUTING"] = "1" # 触发 decoding.py 中的路由逻辑
        
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

def _release_gpu_cache():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

def main():
    args = _parse_args()
    
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("data", data)[:args.limit] if args.limit else data.get("data", data)

    home_dir = Path("/tmp/openeqa_routing_ablation")
    home_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(home_dir)
    
    results = []
    print(f"🚀 开始消融实验 (Baseline vs Vanilla SD vs Routed SD)，共 {len(samples)} 个样本")
    
    # 初始化一个共用的 Client 以加载模型
    llm_config = LLMConfig(
        model="qwen3-vl-speculative",
        model_endpoint_type="speculative_memory",
        context_window=8192,
        max_tokens=32
    )
    client = LLMClient.create(llm_config=llm_config)

    for idx, sample in enumerate(samples):
        question = sample.get("question", "")
        image_paths = sample.get("image_paths") or sample.get("images") or []
        print(f"\n[{idx+1}/{len(samples)}] Q: {question[:60]}...")
        
        # 1. 借助 Agent 框架产出记忆 (不计入测速)
        wipe_mma_sqlite(home_dir)
        agent_wrapper = _init_agent(args.config, for_qa=False)
        mma_agent = agent_wrapper.agent
        ensure_episodic_from_frames(mma_agent, image_paths, sample)
        
        all_events = list_reranked_episodic_for_question(mma_agent, question)
        selected_events = select_events_for_qa(all_events, question)
        memory_items = events_to_memory_items(selected_events)
        
        prepare_draft_policy_for_agent(mma_agent, question)
        llm_config.max_tokens = 8 if is_yes_no_question(question) else 32
        
        raw_prompt = _format_eqa_question(question)
        block = format_episodic_block_for_qa(selected_events)
        prompt_with_memory = f"Episodic Memory:\n{block}\n\n{raw_prompt}"
        
        request_data = client.build_request_data([], llm_config)
        request_data["chat"] = [{"role": "user", "content": prompt_with_memory}]
        request_data["memory_items"] = memory_items
        request_data["local_rag"] = False
        request_data["max_new_tokens"] = llm_config.max_tokens
        
        sample_result = {"question": question, "gold": sample.get("answer")}
        
        # ================= 核心测速 (完全脱离 Agent) =================
        modes = ["baseline", "vanilla_sd", "routed_sd"]
        for mode in modes:
            _release_gpu_cache()
            try:
                with _ablation_env(mode):
                    res = client.request(dict(request_data, collect_stats=(mode != "baseline"), stats_out={}))
                
                t_sec = res.get("elapsed_sec", 0.0)
                pred = res.get("generated_text", "").strip()
                stats = res.get("speculative_stats", {})
                acc_rate = stats.get("acceptance_rate", 0.0)
                mem_after_routing = stats.get("memories_after_routing", len(memory_items))
                
                print(f"  [{mode:10s}] 耗时: {t_sec:.3f}s | Acc: {acc_rate:.2f} | Mems: {mem_after_routing} | 预测: {pred[:30]!r}")
                
                sample_result[f"{mode}_sec"] = t_sec
                sample_result[f"{mode}_pred"] = pred
                sample_result[f"{mode}_acc"] = acc_rate
            except Exception as e:
                print(f"  ❌ {mode} 失败: {e}")
                sample_result[f"{mode}_sec"] = 0.0
        # =============================================================
        
        # 计算加速比
        if sample_result.get("vanilla_sd_sec") and sample_result.get("routed_sd_sec"):
            t_base = sample_result["baseline_sec"]
            sample_result["speedup_vanilla"] = t_base / sample_result["vanilla_sd_sec"]
            sample_result["speedup_routed"]  = t_base / sample_result["routed_sd_sec"]
            print(f"  => Vanilla 加速比: {sample_result['speedup_vanilla']:.3f}x | Routed 加速比: {sample_result['speedup_routed']:.3f}x")
            
        results.append(sample_result)

    # 统计汇总
    valid_res = [r for r in results if r.get("speedup_vanilla")]
    if valid_res:
        summary = {
            "mean_vanilla_speedup": sum(r["speedup_vanilla"] for r in valid_res) / len(valid_res),
            "mean_routed_speedup": sum(r["speedup_routed"] for r in valid_res) / len(valid_res),
            "mean_vanilla_acc": sum(r["vanilla_sd_acc"] for r in valid_res) / len(valid_res),
            "mean_routed_acc": sum(r["routed_sd_acc"] for r in valid_res) / len(valid_res)
        }
        print("\n✅ 消融实验完成！")
        print(f"  Vanilla SD 平均加速比: {summary['mean_vanilla_speedup']:.3f}x (接受率: {summary['mean_vanilla_acc']:.3f})")
        print(f"  Routed SD  平均加速比: {summary['mean_routed_speedup']:.3f}x (接受率: {summary['mean_routed_acc']:.3f})")
        
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()