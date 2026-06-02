"""MLP Manifold Projector — slim canonical estimator for src.

Plays three roles in the canonical pipeline:
  - C1 (safety filter):    "is this patch on the normal manifold?"
  - C2 (zone classifier):  "Zone 1 (dominant) / Zone 2 (residual) / Zone 3 (reject)?"
  - C3 (gate signal):      "should residual correction be applied?"

A single instance is fitted once on M₀ features and reused for all three roles.

Deferred extensions (refit / online calibration / spatial filter / density-aware
threshold / adaptive cumulative tau) live in:
  experimental/mlp_refit/manifold_refit_estimator.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Bottleneck autoencoder (FoundAD-inspired)
# =============================================================================
class MLPProjector(nn.Module):
    """Bottleneck autoencoder for normal-manifold reconstruction.

    Architecture:
        Input(D) -> Linear(D, D//r) -> GELU -> Linear(D//r, D//r) -> GELU -> Linear(D//r, D)

    Anomalous patches reconstruct poorly → high MSE serves as anomaly signal.
    """

    def __init__(self, feat_dim: int, bottleneck_ratio: int = 4):
        super().__init__()
        hidden = feat_dim // bottleneck_ratio
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit(
        self,
        train_feats: np.ndarray,
        n_epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 512,
        device: str = "cuda:0",
    ) -> dict:
        """Train projector on normal features (clean reconstruction).

        Returns:
            {"train_mse_mean": float, "train_mse_std": float}
        """
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(dev)
        self.train()

        X = torch.from_numpy(train_feats.astype(np.float32)).to(dev)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)

        n = len(X)
        for _epoch in range(n_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, batch_size):
                batch = X[perm[i:i + batch_size]]
                pred = self(batch)
                loss = nn.functional.mse_loss(pred, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Train statistics
        self.eval()
        with torch.no_grad():
            pred_all = self(X)
            errors = ((X - pred_all) ** 2).mean(dim=1)
            return {
                "train_mse_mean": float(errors.mean()),
                "train_mse_std": float(errors.std()),
            }

    @torch.no_grad()
    def compute_projection_error(self, feats: np.ndarray, device: str = "cuda:0") -> np.ndarray:
        """Per-patch reconstruction MSE: ||x - MLP(x)||² averaged over feature dim.

        Returns:
            errors: (N,) float32 array (NaN/inf-safe — caller must handle).
        """
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(dev)
        self.eval()
        X = torch.from_numpy(feats.astype(np.float32)).to(dev)
        pred = self(X)
        errors = ((X - pred) ** 2).mean(dim=1)
        return errors.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def reconstruct(self, feats: np.ndarray, device: str = "cuda:0") -> np.ndarray:
        """Per-patch MLP reconstruction MLP(x) = decode(encode(x)).

        Returns:
            recon: (N, D) float32 array — the on-manifold reconstruction.
        """
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(dev)
        self.eval()
        X = torch.from_numpy(feats.astype(np.float32)).to(dev)
        pred = self(X)
        return pred.cpu().numpy().astype(np.float32)


# =============================================================================
# SubspaceEstimator — minimal wrapper around MLPProjector
# =============================================================================
@dataclass
class SubspaceEstimator:
    """Canonical estimator: MLP manifold projector trained once on M₀.

    Used by the canonical pipeline (src/pipeline/) for all three roles
    (safety filter, zone classifier, gate signal). All call sites read
    `_train_error_mean` (after `fit`) and call `_compute_projection_error`.

    The protected names are kept for backward compatibility with existing
    analysis/measure_*/phase*/rq* scripts that import this class directly.
    """

    manifold_bottleneck_ratio: int = 4
    manifold_n_epochs: int = 200
    manifold_lr: float = 1e-3

    # Internal state (set after fit)
    _projector: Optional[MLPProjector] = field(default=None, repr=False)
    _train_error_mean: float = field(default=0.0, repr=False)
    _train_error_std: float = field(default=0.0, repr=False)
    _is_fitted: bool = field(default=False, repr=False)
    _device: str = field(default="cuda:0", repr=False)

    def fit(self, train_feats: np.ndarray, device: str = "cuda:0") -> "SubspaceEstimator":
        """Train MLP manifold projector on normal features."""
        train_feats = np.asarray(train_feats, dtype=np.float32)
        self._device = device

        feat_dim = train_feats.shape[1]
        self._projector = MLPProjector(feat_dim, self.manifold_bottleneck_ratio)
        stats = self._projector.fit(
            train_feats,
            n_epochs=self.manifold_n_epochs,
            lr=self.manifold_lr,
            device=device,
        )
        self._train_error_mean = stats["train_mse_mean"]
        self._train_error_std = stats["train_mse_std"]
        self._is_fitted = True
        return self

    def _compute_projection_error(self, X: np.ndarray) -> np.ndarray:
        """Compute MLP reconstruction error per patch.

        This is the single signal that drives all three roles
        (filter / zone classifier / gate). Lower = more normal.
        """
        if not self._is_fitted:
            raise RuntimeError("SubspaceEstimator not fitted. Call .fit() first.")
        return self._projector.compute_projection_error(X, device=self._device)

    # ---- Public aliases (preferred for new code) ----
    @property
    def train_error_mean(self) -> float:
        return self._train_error_mean

    def project_error(self, X: np.ndarray) -> np.ndarray:
        return self._compute_projection_error(X)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """MLP reconstruction MLP(x) per patch — (N, D) float32.

        Read by the Cycle 9 reconstruction-aware scoring probe (arms B / B2);
        the canonical pipeline never calls this. Additive — no canonical-path
        behaviour change.
        """
        if not self._is_fitted:
            raise RuntimeError("SubspaceEstimator not fitted. Call .fit() first.")
        return self._projector.reconstruct(X, device=self._device)
