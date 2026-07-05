import torch
import torch.nn.functional as F
from typing import List, Tuple, Any

def route_text_memory_items(
    memory_items: List[Any], 
    query_ids: torch.Tensor, 
    target_model: Any, 
    tokenizer: Any, 
    budget: int,
    device: torch.device
) -> List[Any]:
    """
    【为 Text Memory 设计的 Item-level Routing】
    通过 Semantic Affinity (Cosine) 给整段文本打分
    """
    if not memory_items or budget <= 0:
        return []
        
    with torch.no_grad():
        embed_layer = target_model.get_input_embeddings()
        
        # Query Embedding: (1, hidden_dim)
        q_emb = embed_layer(query_ids.to(device)).mean(dim=1)
        q_emb = F.normalize(q_emb, p=2, dim=-1)
        
        item_scores = []
        item_lengths = []
        
        # 对每一个 Memory Item 计算 Semantic Affinity
        for item in memory_items:
            content = item.get("content", "") if isinstance(item, dict) else getattr(item, "content", "")
            if not content:
                item_scores.append(-1.0)
                item_lengths.append(0)
                continue
                
            i_ids = tokenizer.encode(content, add_special_tokens=False)
            item_lengths.append(len(i_ids))
            
            if len(i_ids) == 0:
                item_scores.append(-1.0)
                continue
                
            i_tensor = torch.tensor([i_ids], device=device)
            # Item Embedding: (1, hidden_dim)
            i_emb = embed_layer(i_tensor).mean(dim=1)
            i_emb = F.normalize(i_emb, p=2, dim=-1)
            
            # Semantic Affinity Score
            score = torch.sum(q_emb * i_emb).item()
            item_scores.append(score)
            
    # 按照分数降序排序
    sorted_indices = sorted(range(len(memory_items)), key=lambda k: item_scores[k], reverse=True)
    
    # 选取 Top Items 直到满足 Budget
    selected_items = []
    current_tokens = 0
    
    for idx in sorted_indices:
        if item_scores[idx] == -1.0:
            continue
        # 保证至少选一个，且不超过预算
        if current_tokens + item_lengths[idx] > budget and current_tokens > 0:
            break
        selected_items.append(memory_items[idx])
        current_tokens += item_lengths[idx]
        
    return selected_items


def route_visual_kv_cache(
    memory_kv_raw: List[Tuple[torch.Tensor, torch.Tensor]],
    query_embeds: torch.Tensor,
    visual_embeds: torch.Tensor,
    budget: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    在 Vision Feature 层级进行细粒度 Patch 截断
    通过 torch.sort 保留 RoPE 空间位置的绝对秩序
    """
    memory_len = memory_kv_raw[0][0].size(2)
    if memory_len <= budget:
        return memory_kv_raw
        
    q_vec = F.normalize(query_embeds.mean(dim=1), p=2, dim=-1)
    m_vecs = F.normalize(visual_embeds, p=2, dim=-1)
    
    # (1, memory_len)
    scores = torch.bmm(q_vec.unsqueeze(1), m_vecs.transpose(1, 2)).squeeze(0).squeeze(0)
    
    # 选出 Top-K，并【严格排序】以防打碎 RoPE 和 Patch 邻接关系
    _, top_indices = torch.topk(scores, budget)
    top_indices, _ = torch.sort(top_indices)
    
    allocated_kv = []
    for k, v in memory_kv_raw:
        new_k = k.index_select(2, top_indices)
        new_v = v.index_select(2, top_indices)
        allocated_kv.append((new_k, new_v))
        
    return allocated_kv