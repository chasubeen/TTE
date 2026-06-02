"""Backward-compatibility shim for `utils.estimator`.

The slim canonical implementation lives in `src.estimator.mlp_projector`.
Older scripts that import `from src.utils.estimator import SubspaceEstimator`
or `MLPProjector` continue to work via this shim.
"""
from src.estimator.mlp_projector import MLPProjector, SubspaceEstimator

__all__ = ["MLPProjector", "SubspaceEstimator"]
