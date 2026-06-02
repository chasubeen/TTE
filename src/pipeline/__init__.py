"""src pipeline: selective test-time memory expansion + single-memory scoring.

Storyline:
  Req-A — Gap-filling selective update           → expansion.py
  Req-B — Memory-independent MLP normality       → estimator/
  Req-C — Ranking-Preserving Adaptation (RPA)    → scoring.py
          (score_baseline: score(q) = d(q, M))

Configuration is loaded from `src/configs/*.yaml` via `src.configs`.
"""
from src.configs import (
    BACKBONE,
    IMG_SIZE,
    RESIZE_MASK,
    ROTATION_ANGLES,
    METRICS,
    DATASET_CATEGORIES,
    DEFAULT_CFG,
    GATE_Q_HIGH,
    GATE_Q_LOW,
    MEMORY_BANK_KWARGS,
    RESIDUAL_GAMMA,
)

__all__ = [
    "BACKBONE",
    "IMG_SIZE",
    "RESIZE_MASK",
    "ROTATION_ANGLES",
    "METRICS",
    "DATASET_CATEGORIES",
    "DEFAULT_CFG",
    "GATE_Q_HIGH",
    "GATE_Q_LOW",
    "MEMORY_BANK_KWARGS",
    "RESIDUAL_GAMMA",
]
