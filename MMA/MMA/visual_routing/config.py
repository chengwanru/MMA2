import os
from dataclasses import dataclass, field


@dataclass
class VisualRoutingConfig:
    enable_routing: bool = field(
        default_factory=lambda: os.environ.get("MMA_ENABLE_VISUAL_ROUTING", "0") == "1"
    )
    # Was hardcoded False with no wiring to ever enable it -- get_evidence_requirement()'s
    # per-question-type budget (256 for color/yes-no, 1024 for detail/describe/why, 512
    # otherwise) was dead code. Add an env override so it can actually be turned on.
    is_dynamic_budget: bool = field(
        default_factory=lambda: os.environ.get("MMA_VISUAL_ROUTING_DYNAMIC_BUDGET", "0") == "1"
    )
    fixed_budget: int = 512
    
    min_budget: int = 128
    max_budget: int = 2048
    
    scoring_strategy: str = "semantic_affinity"