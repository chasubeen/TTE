"""Distance scorer (Req-C) — the canonical single-reference baseline.

    anomaly_score(q) = d(q, M)

where M is the working memory (M₀ + safely-absorbed patches) owned by the
MemoryExpander. This is the AnomalyDINO-style 1-NN distance score.
"""
import numpy as np

from src.scorer.common import _post_process_to_score_map, _top1_percent_mean


def score_baseline(feats_np, bank, spatial_shape, category, dataset):
    """Canonical scorer: anomaly_score(q) = d(q, M)."""
    d, _ = bank.query(feats_np, k=1)
    patch_scores = d[:, 0].astype(np.float32)
    score_map, patch_scores = _post_process_to_score_map(
        patch_scores, feats_np, spatial_shape, category, dataset)
    return score_map, _top1_percent_mean(patch_scores)
