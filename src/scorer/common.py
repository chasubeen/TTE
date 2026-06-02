"""Shared scoring post-processing (FG mask → reshape → smooth → top-1%).

Every scorer returns (score_map, image_score):
  score_map   : (RESIZE_MASK, RESIZE_MASK) Gaussian-smoothed pixel anomaly map
  image_score : top-1% mean over (foreground-masked) patch scores
"""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from src.utils.augmentation import compute_foreground_mask, get_masking_default
from src.configs import RESIZE_MASK


def _post_process_to_score_map(patch_scores, feats_np, spatial_shape,
                               category, dataset):
    """FG masking → reshape → resize → Gaussian smooth."""
    H, W = spatial_shape
    if get_masking_default(dataset, category):
        fg = compute_foreground_mask(feats_np, (H, W), threshold=10.0, kernel_size=3)
        patch_scores = np.where(fg, patch_scores, 0.0)

    score_map = patch_scores.reshape(H, W)
    score_map = cv2.resize(score_map, (RESIZE_MASK, RESIZE_MASK),
                           interpolation=cv2.INTER_LINEAR)
    score_map = gaussian_filter(score_map, sigma=4)
    return score_map, patch_scores


def _top1_percent_mean(patch_scores):
    """Top-1% mean (AnomalyDINO image-level aggregation)."""
    k = max(1, int(len(patch_scores) * 0.01))
    return float(np.sort(patch_scores)[-k:].mean())
