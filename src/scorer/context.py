"""Context-signal scorer (Req-C / Ranking-Preserving Adaptation).

An exogenous (memory-INDEPENDENT) context signal g(q) discounts the distance
score so normal-like patches are suppressed more than anomalous ones, preserving
the normal–anomaly ranking after expansion:

    s'(q) = d(q, M) · (1 − λ · g(q))

g(q) ∈ [0,1] measures how normal-like q is relative to OTHER patches:
  S4 (cross-batch, canonical): mean top-k cosine to other-image FG patches.
  S3 (intra-image rank):       1 − rank(local kNN distance) within the image.
Canonical method uses S4 with λ=0.5, k=5.
"""
import numpy as np

from src.scorer.common import _post_process_to_score_map, _top1_percent_mean


def compute_context_g(feats_np, fg_idx, k, signal,
                      batch_pool_norm=None, batch_pool_img=None, img_idx=0):
    """Per-patch context signal g(q) ∈ [0,1] (length = len(feats_np)).

    Non-foreground patches and (when too few neighbours exist) all patches get
    g = 0 (no discount). `signal` ∈ {"S3", "S4"}.  S4 needs the batch FG pool:
      - `batch_pool_norm` : (P, D) L2-normalised FG features of the whole batch
      - `batch_pool_img`  : (P,) source-image index for each pooled patch
      - `img_idx`         : this image's index (its own patches are excluded)
    """
    N_total = len(feats_np)
    g = np.zeros(N_total, dtype=np.float32)
    if len(fg_idx) <= k + 1:
        return g
    F_fg = feats_np[fg_idx].astype(np.float32)
    F_norm = F_fg / (np.linalg.norm(F_fg, axis=1, keepdims=True) + 1e-8)

    if signal == "S3":
        cos_mat = F_norm @ F_norm.T
        np.fill_diagonal(cos_mat, -np.inf)
        top_k = np.sort(cos_mat, axis=1)[:, -k:]
        d_local = np.sqrt(np.maximum(2.0 - 2.0 * top_k, 0.0).mean(axis=1))
        ranks = np.argsort(np.argsort(d_local))
        g[fg_idx] = 1.0 - ranks.astype(np.float32) / max(len(fg_idx) - 1, 1)
    elif signal == "S4":
        if batch_pool_norm is None or batch_pool_norm.shape[0] == 0:
            return g
        cos = F_norm @ batch_pool_norm.T
        same_img = (batch_pool_img == img_idx)
        cos[:, same_img] = -np.inf
        if int((~same_img).sum()) >= k:
            top_k = np.sort(cos, axis=1)[:, -k:]
            g[fg_idx] = top_k.mean(axis=1).clip(0.0, 1.0)
    else:
        raise ValueError(
            f"Unknown context signal {signal!r}; expected 'S3' or 'S4'.")
    return g


def score_context_discount(feats_np, bank, g, lam, spatial_shape, category, dataset):
    """Ranking-Preserving Adaptation scorer: s'(q) = d(q, M)·(1 − λ·g(q)).

    λ=0 reduces to `score_baseline`. Post-processing is identical to every scorer.
    """
    d, _ = bank.query(feats_np, k=1)
    s_raw = d[:, 0].astype(np.float32)
    patch_scores = (s_raw * (1.0 - float(lam) * np.asarray(g, dtype=np.float32))
                    ).astype(np.float32)
    score_map, patch_scores = _post_process_to_score_map(
        patch_scores, feats_np, spatial_shape, category, dataset)
    return score_map, _top1_percent_mean(patch_scores)
