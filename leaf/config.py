"""Fixed main-method configuration."""
from dataclasses import dataclass

CLASSIFICATION_DATASETS = (
    "adult", "blood", "breast-w", "credit-g", "cdc_diabetes", "heart", "gallstone",
)
REGRESSION_DATASETS = (
    "cpu_small", "diamonds", "forest-fires", "housing", "bike",
    "cultivars_regression", "infrared_thermography_temperature", "wine", "insurance",
)

@dataclass(frozen=True)
class LEAFConfig:
    n_estimators: int = 9
    max_depth: int = 4
    max_features: float = 0.9
    max_derived_features: int = 8
    max_candidates_per_feature: int = 5
    max_candidates_per_node: int = 64
    min_labeled_leaf: int = 2
    min_labeled_child: int = 1
    min_total_child: int = 2
    lambda_lab: float = 1.0
    lambda_tea: float = 1.0
    lambda_str: float = 0.3
    eta: float = 4.0
    kappa: float = 2.0
    gamma: float = 1.0

@dataclass(frozen=True)
class LLMConfig:
    model: str = "gpt-5.4-nano"
    backend: str = "api"
    max_new_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 1.0
    cache_dir: str = "outputs/llm_cache"
    device: str = "auto"
