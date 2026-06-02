"""Slim canonical evaluation metrics for src.

Provides:
  - ader_evaluator: full 7-metric evaluation via the `adeval` library
  - f1_score_max: helper used by ader_evaluator

Legacy helpers (memory_patch_scores, penalty_patch_scores, dual_memory_patch_scores,
scores_list_to_maps, image_level_scores_from_maps, resize_gt_masks, get_logger,
setup_seed) are deprecated — see experimental/legacy_scripts/metrics_legacy.py
if reproducing src10/src11-era experiments requires them.
"""
import numpy as np
import torch
from typing import Sequence

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from adeval import EvalAccumulatorCuda


def f1_score_max(y_true, y_score) -> float:
    """Maximum F1 score over the precision-recall curve."""
    precs, recs, _ = precision_recall_curve(y_true, y_score)
    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return float(f1s.max())


def ader_evaluator(
    pr_px: np.ndarray,
    pr_sp: np.ndarray,
    gt_px: np.ndarray,
    gt_sp: np.ndarray,
    use_metrics: Sequence[str] = (
        "I-AUROC", "I-AP", "I-F1_max", "P-AUROC", "P-AP", "P-F1_max", "AUPRO",
    ),
):
    """Full 7-metric anomaly detection evaluation via adeval (CUDA required).

    Args:
        pr_px: (N, H, W) pixel-level anomaly maps
        pr_sp: (N,)      image-level anomaly scores
        gt_px: (N, H, W) binary pixel ground truth
        gt_sp: (N,)      binary image labels
        use_metrics: subset/order of metrics to return

    Returns:
        list of metric values in the order requested by `use_metrics`.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for adeval evaluation")

    if pr_px.ndim == 4:
        pr_px = pr_px.squeeze(1)
    if gt_px.ndim == 4:
        gt_px = gt_px.squeeze(1)

    score_min, score_max = float(np.min(pr_sp)), float(np.max(pr_sp))
    anomap_min, anomap_max = float(pr_px.min()), float(pr_px.max())

    # Guard against degenerate score ranges (e.g. rank-based scoring saturating
    # at max(D₀) for highly anomalous categories).  adeval's EvalAccumulatorCuda
    # asserts estimated_score_lower < estimated_score_upper; widening by a tiny
    # epsilon preserves ranking-based AUROC while preventing the assertion.
    _eps = 1e-6
    if score_max - score_min < _eps:
        score_max = score_min + _eps
    if anomap_max - anomap_min < _eps:
        anomap_max = anomap_min + _eps

    accum = EvalAccumulatorCuda(
        score_min, score_max, anomap_min, anomap_max,
        skip_pixel_aupro=False, nstrips=200,
    )
    accum.add_anomap_batch(
        torch.tensor(pr_px).cuda(non_blocking=True),
        torch.tensor(gt_px.astype(np.uint8)).cuda(non_blocking=True),
    )

    metrics = accum.summary()
    out = {}
    for metric in use_metrics:
        if metric == "I-AUROC":
            out[metric] = roc_auc_score(gt_sp, pr_sp)
        elif metric == "I-AP":
            out[metric] = average_precision_score(gt_sp, pr_sp)
        elif metric == "I-F1_max":
            out[metric] = f1_score_max(gt_sp, pr_sp)
        elif metric == "P-AUROC":
            out[metric] = metrics["p_auroc"]
        elif metric == "P-AP":
            out[metric] = metrics["p_aupr"]
        elif metric == "P-F1_max":
            out[metric] = f1_score_max(gt_px.ravel(), pr_px.ravel())
        elif metric == "AUPRO":
            out[metric] = metrics["p_aupro"]

    return [out[m] for m in use_metrics]
