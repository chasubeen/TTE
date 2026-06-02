"""Backward-compatibility shim — moved to the `src.scorer` package.

The canonical scorers now live in:
  - `src.scorer.distance`  : score_baseline  (Req-C, d(q, M))
  - `src.scorer.context`   : compute_context_g, score_context_discount
                               (Ranking-Preserving Adaptation)
  - `src.scorer.common`    : shared post-processing helpers

The graveyard scorers (RPA blend/2-sided, gated-dual, M₀-anchored rank,
confidence-weighted/influence-discounted, proto-density, SoftPatch/max-anchor,
Cycle-9 reconstruction) were removed — they are all refuted (see docs §5 / the
A6 closure). Existing imports of the *live* scorers keep working.
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
