"""Runner orchestration for the canonical batch-by-batch pipeline.

Provides:
  evaluate_category: one (category, seed) → metrics dict
  run: top-level entry that loops seeds × categories, aggregates, saves JSON

This is the high-level glue. The scientific logic lives in the functional
packages it wires together:
  - selector/  (Req-B: MLP gate — confident-normal selection)
  - memory/    (Req-A: M₀ build + selective expansion, reservoir/append)
  - scorer/    (Req-C: distance score + Ranking-Preserving context discount)
"""
import csv
import json
import logging
import os
import random
import traceback
import warnings
from pathlib import Path

# Deterministic cuBLAS — must be set before the first CUDA op (CUDA context is
# created in run()/evaluate_category, which run after this module import).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch

# Silence only the (harmless) non-deterministic-op warnings raised by the adeval
# metric library (histc/cumsum on CUDA) under warn_only=True. These ops live in
# evaluation, not the method, and do not change reported metrics at 4-dp; strict
# mode would crash on them, so warn_only=True is intentional. Other warnings are
# left visible.
warnings.filterwarnings(
    "ignore",
    message=r".*does not have a deterministic implementation",
    category=UserWarning,
)
# DINOv2 attention falls back to standard (non-xFormers) ops when xFormers is
# not installed — identical numerics, just slower. Silence the startup notice.
warnings.filterwarnings("ignore", message=r".*xFormers is not available.*",
                        category=UserWarning)
# 0/0 divides in the adeval metric library (TPR/FPR at degenerate thresholds) —
# harmless and handled. Scoped to adeval so real divides in the method still warn.
warnings.filterwarnings("ignore", message=r".*invalid value encountered in divide",
                        category=RuntimeWarning, module=r"adeval.*")


def enable_determinism(seed: int) -> None:
    """Seed all RNGs + enable deterministic algorithms (reproducibility).

    Guarantees bit-identical runs on the SAME machine + torch/CUDA/cuDNN stack.
    Cross-machine reproduction of the MLP-gated expansion additionally requires
    an identical GPU/cuDNN stack (the static 1-NN baseline is already machine-
    portable). `warn_only=True` keeps the run from crashing if some op lacks a
    deterministic implementation; flip to False to hard-enforce.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

from src.models.patch_extractor import PatchFeatureExtractor
from src.selector import SubspaceEstimator
from src.utils.metrics import ader_evaluator

from src.configs import (
    BACKBONE, IMG_SIZE, RESIZE_MASK, METRICS, DATASET_CATEGORIES,
    DEFAULT_CFG, MLP_CFG,
)
from src.utils.augmentation import compute_foreground_mask, get_masking_default
from src.memory.build import build_memory, load_test_images
from src.memory.expander import MemoryExpander
from src.scorer import score_baseline, compute_context_g, score_context_discount

logger = logging.getLogger(__name__)


def _gt_to_binary(gt_mask):
    """Resize ground-truth mask to RESIZE_MASK and binarize."""
    gt_r = cv2.resize(gt_mask.astype(np.float32),
                      (RESIZE_MASK, RESIZE_MASK), interpolation=cv2.INTER_NEAREST)
    return (gt_r > 0.5).astype(np.uint8)


def _evaluate_baseline(images, bank, spatial_shape, category, dataset):
    """Score every test image with the static baseline (no expansion): d(q, M₀)."""
    px, sp, gt_px, gt_sp = [], [], [], []
    for img_data in images:
        sm, isc = score_baseline(
            img_data["feats"], bank, spatial_shape, category, dataset)
        px.append(sm); sp.append(isc)
        gt_px.append(_gt_to_binary(img_data["gt_mask"]))
        gt_sp.append(img_data["label"])
    m = ader_evaluator(np.stack(px), np.array(sp), np.stack(gt_px), np.array(gt_sp))
    return {METRICS[i]: m[i] for i in range(7)}


def _evaluate_ours(images, bank_M0, bank, estimator, spatial_shape, category, dataset, cfg, device,
                   scoring_mode_override=None, D0_sorted=None):
    """Run canonical batch-by-batch selective expansion on the test stream.

    Supports three scoring modes (via cfg["scoring_mode"] or override):
      - "baseline"           : score_baseline = d(q, M_t).  B arm.
      - "m0_anchored_rank"   : score_m0_anchored_rank with D₀ reference.  D arm.
      - "gated_dual"         : legacy ablation (kept for reproducibility).

    bank_M0: passed for backward compat; the simplified pipeline does not use a
             separate M₀ reference at score time (D₀ is the only M₀-derived
             quantity needed at score time, supplied via D0_sorted).
    bank:    initial memory bank handed to SelectiveExpander (copied internally).
    """
    expander = MemoryExpander(bank, estimator, cfg, device)
    bs = cfg["batch_size"]
    px, sp, gt_px, gt_sp = [], [], [], []
    n_batches = (len(images) + bs - 1) // bs

    for bi in range(n_batches):
        batch_imgs = images[bi * bs : (bi + 1) * bs]

        # ---- Score this batch with the CURRENT memory state ----
        for img_data in batch_imgs:
            sm, isc = score_baseline(
                img_data["feats"], expander.bank, spatial_shape, category, dataset)
            px.append(sm); sp.append(isc)
            gt_px.append(_gt_to_binary(img_data["gt_mask"]))
            gt_sp.append(img_data["label"])

        # ---- Then absorb this batch into the expanded memory ----
        batch_feats = np.concatenate([img["feats"] for img in batch_imgs], axis=0)
        expander.absorb_batch(batch_feats)

    m = ader_evaluator(np.stack(px), np.array(sp), np.stack(gt_px), np.array(gt_sp))
    return {METRICS[i]: m[i] for i in range(7)}, expander.stats()


def _evaluate_context_current(images, bank, estimator, spatial_shape, category, dataset,
                              cfg, device):
    """Current method — reservoir memory + Ranking-Preserving Adaptation.

    For each batch (score-then-expand): build the batch foreground pool, compute
    the exogenous context signal g(q), score s'(q) = d(q, M)·(1 − λ·g(q)) against
    the current (pre-absorb) memory, then absorb via the reservoir policy.
    Mirrors analysis.context_robust_validation.run_trial.  Activated by
    cfg["scoring_mode"] == "context_discount" (with memory_policy: reservoir).
    """
    expander = MemoryExpander(bank, estimator, cfg, device)
    bs = cfg["batch_size"]
    signal = cfg.get("context_signal", "S4")
    lam = float(cfg.get("context_lambda", 0.5))
    k_ctx = int(cfg.get("context_k", 5))
    do_fg = get_masking_default(dataset, category)
    H, W = spatial_shape
    dim = bank.features.shape[1]

    px, sp, gt_px, gt_sp = [], [], [], []
    n_batches = (len(images) + bs - 1) // bs

    for bi in range(n_batches):
        batch_imgs = images[bi * bs:(bi + 1) * bs]

        # ---- build batch foreground pool for the cross-image (S4) signal ----
        fg_idx_per_img, pool_F, pool_img_idx = [], [], []
        for i, img in enumerate(batch_imgs):
            F = img["feats"].astype(np.float32)
            if do_fg:
                fg_mask = compute_foreground_mask(F, (H, W), threshold=10.0, kernel_size=3)
                fg_idx = np.where(fg_mask)[0]
            else:
                fg_idx = np.arange(len(F))
            fg_idx_per_img.append(fg_idx)
            pool_F.append(F[fg_idx])
            pool_img_idx.extend([i] * len(fg_idx))
        pool_F_np = (np.concatenate(pool_F, axis=0).astype(np.float32)
                     if pool_F else np.empty((0, dim), dtype=np.float32))
        pool_img_np = np.array(pool_img_idx, dtype=np.int32)
        pool_norm = (pool_F_np / (np.linalg.norm(pool_F_np, axis=1, keepdims=True) + 1e-8)
                     if len(pool_F_np) else None)

        # ---- score this batch with the CURRENT memory (before absorb) ----
        for i, (img_data, fg_idx) in enumerate(zip(batch_imgs, fg_idx_per_img)):
            F = img_data["feats"].astype(np.float32)
            g = compute_context_g(F, fg_idx, k_ctx, signal,
                                  batch_pool_norm=pool_norm,
                                  batch_pool_img=pool_img_np, img_idx=i)
            sm, isc = score_context_discount(
                F, expander.bank, g, lam, spatial_shape, category, dataset)
            px.append(sm); sp.append(isc)
            gt_px.append(_gt_to_binary(img_data["gt_mask"]))
            gt_sp.append(img_data["label"])

        # ---- then absorb this batch (reservoir) ----
        batch_feats = np.concatenate([im["feats"] for im in batch_imgs], axis=0)
        expander.absorb_batch(batch_feats)

    m = ader_evaluator(np.stack(px), np.array(sp), np.stack(gt_px), np.array(gt_sp))
    metrics = {METRICS[i]: m[i] for i in range(7)}
    return metrics, expander.stats()


def evaluate_category(category, extractor, device, data_path, dataset,
                      shot=1, seed=1, cfg=None):
    """Evaluate one (category, seed): baseline + ours + delta."""
    cfg = cfg or DEFAULT_CFG
    logger.info(f"\n===== {category} (seed={seed}, shot={shot}) =====")
    enable_determinism(seed)

    # ---- Training: M₀ + MLP ----
    bank, spatial_shape = build_memory(
        category, extractor, device, data_path, dataset, shot=shot, seed=seed)
    images = load_test_images(category, extractor, device, data_path, dataset)

    estimator = SubspaceEstimator(
        manifold_bottleneck_ratio=MLP_CFG["bottleneck_ratio"],
        manifold_n_epochs=MLP_CFG["n_epochs"],
        manifold_lr=MLP_CFG["lr"],
    )
    estimator.fit(bank.features, device=device)

    result = {}

    # ---- A arm: Static M₀ + score_baseline ----
    result["baseline"] = _evaluate_baseline(images, bank, spatial_shape, category, dataset)
    bm = result["baseline"]
    logger.info(f"  [A] baseline:    I={bm['I-AUROC']:.4f} P={bm['P-AUROC']:.4f} PRO={bm['AUPRO']:.4f}")

    # ---- B arm: Selective expansion ----
    # Current method (REPORT_20260526): scoring_mode=context_discount runs
    # reservoir + Ranking-Preserving Adaptation.  Default scoring_mode=baseline
    # keeps the legacy gap-priority append-only + score_baseline path.
    if cfg.get("scoring_mode") == "context_discount":
        ours_metrics, ours_meta = _evaluate_context_current(
            images, bank, estimator, spatial_shape, category, dataset, cfg, device)
    else:
        ours_metrics, ours_meta = _evaluate_ours(
            images, bank, bank, estimator, spatial_shape, category, dataset, cfg, device,
            scoring_mode_override="baseline")
    result["ours"] = ours_metrics
    result.update(ours_meta)
    om = result["ours"]
    logger.info(f"  [B] ours:        I={om['I-AUROC']:.4f} P={om['P-AUROC']:.4f} PRO={om['AUPRO']:.4f} "
                f"(mem={ours_meta['mem_expanded']} added={ours_meta['added']})")

    # ---- Delta B−A ----
    for mk in METRICS:
        result[f"d{mk}"] = result["ours"][mk] - result["baseline"][mk]
    logger.info(f"  [B−A] delta:     dI={result['dI-AUROC']:+.4f} dP={result['dP-AUROC']:+.4f} "
                f"dPRO={result['dAUPRO']:+.4f}")

    return result


# ============================================================
# Top-level run + summary
# ============================================================
def _per_seed_summary(seed, sr, categories):
    valid = [c for c in categories if c in sr]
    if not valid:
        return
    logger.info(f"\n--- Seed {seed}: {len(valid)}-cat Mean ---")
    for method in ["baseline", "ours"]:
        means = {m: np.mean([sr[c][method][m] for c in valid]) for m in METRICS}
        logger.info(f"  {method:<10s}: I={means['I-AUROC']:.4f} "
                    f"P={means['P-AUROC']:.4f} PRO={means['AUPRO']:.4f}")
    deltas = {m: np.mean([sr[c][f"d{m}"] for c in valid]) for m in METRICS}
    logger.info(f"  {'delta':<10s}: dI={deltas['I-AUROC']:+.4f} "
                f"dP={deltas['P-AUROC']:+.4f} dPRO={deltas['AUPRO']:+.4f}")


def _multi_seed_aggregate(seeds, all_seed_results, categories, dataset, shot):
    if len(seeds) <= 1:
        return
    all_valid = set(categories)
    for s in seeds:
        all_valid &= set(all_seed_results.get(s, {}).keys())
    valid = [c for c in categories if c in all_valid]

    logger.info(f"\n{'='*70}")
    logger.info(
        f"{len(seeds)}-SEED AGGREGATE "
        f"({dataset}, {shot}-shot, {len(valid)}/{len(categories)}-cat common set)")
    logger.info(f"{'='*70}")
    if not valid:
        logger.warning("No categories succeeded for every seed; skip multi-seed aggregate.")
        return
    dropped = [c for c in categories if c not in all_valid]
    if dropped:
        logger.warning(
            "Excluded categories missing from at least one seed aggregate: "
            + ", ".join(dropped))

    for method in ["baseline", "ours"]:
        seed_means = []
        for s in seeds:
            seed_means.append(
                {m: np.mean([all_seed_results[s][c][method][m] for c in valid])
                 for m in METRICS})
        if seed_means:
            logger.info(f"\n{method}:")
            for m in METRICS:
                vals = [sm[m] for sm in seed_means]
                logger.info(f"  {m:<10s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    logger.info(f"\nDelta (ours - baseline):")
    for m in METRICS:
        delta_vals = []
        for s in seeds:
            delta_vals.append(np.mean([all_seed_results[s][c][f"d{m}"] for c in valid]))
        if delta_vals:
            logger.info(f"  d{m:<10s}: {np.mean(delta_vals):+.4f} ± {np.std(delta_vals):.4f}")


# Paper-table display names: internal key → column header shown in the CSV.
_METRIC_DISPLAY = {
    "I-AUROC":   "I-AUROC",
    "I-AP":      "I-AP",
    "I-F1_max":  "I-F1",
    "P-AUROC":   "P-AUROC",
    "P-AP":      "P-AP",
    "P-F1_max":  "P-F1",
    "AUPRO":     "AUPRO",
}


def _save_paper_csv(out_path, shot, dataset, seeds, all_seed_results, categories):
    """Save paper-table formatted CSV alongside the JSON log.

    Format (matches the image table layout):
      Shot  | Row        | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | AUPRO
      ------+------------+---------+------+------+---------+------+------+------
      1     | Baseline   | 0.9570  | ...  | ...  | ...     | ...  | ...  | ...
      1     | Ours       | 0.9606  | ...  | ...  | ...     | ...  | ...  | ...
      1     | Δ (pp)     | +0.36   | ...  | ...  | ...     | ...  | ...  | ...

    Absolute values: raw [0, 1] float, 4 decimal places.
    Δ row: (Ours − Baseline) × 100 in percentage points, signed, 2 decimal places.
    Means are computed over the intersection of categories that succeeded in
    every seed (same logic as _multi_seed_aggregate).
    """
    valid = set(categories)
    for s in seeds:
        valid &= set(all_seed_results.get(s, {}).keys())
    valid_cats = [c for c in categories if c in valid]
    if not valid_cats:
        logger.warning("_save_paper_csv: no common categories — skipping CSV.")
        return

    def _avg(method, metric):
        return float(np.mean([
            np.mean([all_seed_results[s][c][method][metric] for c in valid_cats])
            for s in seeds
        ]))

    def _avg_delta_pp(metric):
        return float(np.mean([
            np.mean([all_seed_results[s][c][f"d{metric}"] for c in valid_cats])
            for s in seeds
        ])) * 100

    display_cols = [_METRIC_DISPLAY.get(m, m) for m in METRICS]
    fieldnames = ["Shot", "Row"] + display_cols
    rows = [
        {"Shot": shot, "Row": "Baseline",
         **{_METRIC_DISPLAY.get(m, m): f"{_avg('baseline', m):.4f}" for m in METRICS}},
        {"Shot": shot, "Row": "Ours",
         **{_METRIC_DISPLAY.get(m, m): f"{_avg('ours', m):.4f}" for m in METRICS}},
        {"Shot": shot, "Row": "Δ (pp)",
         **{_METRIC_DISPLAY.get(m, m): f"{_avg_delta_pp(m):+.2f}" for m in METRICS}},
    ]

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Paper table CSV saved: {out_path}")


def run(args, cfg=None):
    """Top-level entry: loop seeds × categories, summarize, save JSON + CSV."""
    cfg = dict(DEFAULT_CFG if cfg is None else cfg)
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if args.data_path is None:
        args.data_path = str(project_root / "data")
    if args.categories is None:
        args.categories = DATASET_CATEGORIES.get(args.dataset, [])

    extractor = PatchFeatureExtractor(backbone_name=BACKBONE, img_size=IMG_SIZE).to(args.device)

    all_seed_results = {}
    for seed in args.seeds:
        logger.info(f"\n{'#'*70}\n# SEED={seed}  SHOT={args.shot}\n{'#'*70}")
        seed_results = {}
        for cat in args.categories:
            try:
                seed_results[cat] = evaluate_category(
                    cat, extractor, args.device, args.data_path, args.dataset,
                    shot=args.shot, seed=seed, cfg=cfg,
                )
            except Exception as e:
                logger.error(f"Failed {cat}: {e}")
                traceback.print_exc()
        all_seed_results[seed] = seed_results

    # Per-seed + multi-seed summary
    for seed in args.seeds:
        _per_seed_summary(seed, all_seed_results[seed], args.categories)
    _multi_seed_aggregate(args.seeds, all_seed_results, args.categories, args.dataset, args.shot)

    # Save JSON
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = str(cfg.get("scoring_mode", "default"))
    novelty_tag = str(cfg.get("novelty_ratio", "na")).replace(".", "p")
    out_name = (
        f"main_table_{args.dataset}_k{args.shot}_s{'_'.join(map(str, args.seeds))}"
        f"_{mode_tag}_nov{novelty_tag}.json"
    )
    payload = {"config": vars(args), "pipeline_cfg": cfg}
    for s in args.seeds:
        payload[f"seed_{s}"] = all_seed_results[s]
    (log_dir / out_name).write_text(json.dumps(payload, indent=2, default=float))
    logger.info(f"\nSaved: {log_dir / out_name}")

    csv_name = out_name.replace(".json", ".csv")
    _save_paper_csv(
        log_dir / csv_name, args.shot, args.dataset,
        args.seeds, all_seed_results, args.categories,
    )
