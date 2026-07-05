def get_evidence_requirement(query_text: str, config) -> int:
    """
    Step 1: Evidence Requirement Estimation
    决定当前问题需要多少 Budget
    """
    if not config.is_dynamic_budget:
        return config.fixed_budget
        
    query_lower = query_text.lower()
    if any(w in query_lower for w in ["detail", "describe", "why"]):
        return min(1024, config.max_budget)
    elif any(w in query_lower for w in ["what color", "yes or no"]):
        return max(256, config.min_budget)
    else:
        return 512