"""Confident-normal selection gate (Req-B / Selection — Memory-independent Gate).

The Selector wraps the M₀-trained frozen MLP and turns its per-patch
reconstruction error into the binary "confident-normal" decision used to choose
which test-stream patches are eligible for memory expansion:

    confident_normal(p)  ⇔  e(p) = ‖MLP(p) − p‖²  <  τ_low

τ_low is anchored to M₀ (τ_low = mean train error × tau_ratio), so the gate is
*memory-independent*: it never drifts as the memory expands.
"""
import numpy as np

from src.selector.mlp_gate import SubspaceEstimator


class Selector:
    """Frozen MLP gate that selects confident-normal patches for expansion."""

    def __init__(self, estimator: SubspaceEstimator, tau_ratio: float):
        self.estimator = estimator
        self.tau_low = estimator._train_error_mean * tau_ratio

    def reconstruction_error(self, feats: np.ndarray) -> np.ndarray:
        """Per-patch MLP reconstruction error e(p)."""
        return self.estimator._compute_projection_error(feats)

    def confident_normal_mask(self, feats: np.ndarray) -> np.ndarray:
        """Boolean mask: True where e(p) < τ_low (admit for expansion)."""
        return self.reconstruction_error(feats) < self.tau_low
