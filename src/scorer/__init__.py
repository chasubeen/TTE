"""scorer — anomaly scoring (Req-C).

  - `score_baseline`          : distance score d(q, M)  (canonical baseline)
  - `compute_context_g`       : exogenous context signal g(q) ∈ [0,1]
  - `score_context_discount`  : Ranking-Preserving Adaptation s'·(1 − λ·g)
"""
from src.scorer.common import _post_process_to_score_map, _top1_percent_mean
from src.scorer.distance import score_baseline
from src.scorer.context import compute_context_g, score_context_discount

__all__ = [
    "score_baseline",
    "compute_context_g",
    "score_context_discount",
    "_post_process_to_score_map",
    "_top1_percent_mean",
]
