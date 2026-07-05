from dataclasses import dataclass

@dataclass
class VisualRoutingConfig:
    enable_routing: bool = field(
        default_factory=lambda: os.environ.get("MMA_ENABLE_VISUAL_ROUTING", "0") == "1"
    )
    is_dynamic_budget: bool = False 
    fixed_budget: int = 512
    
    min_budget: int = 128
    max_budget: int = 2048
    
    scoring_strategy: str = "semantic_affinity"