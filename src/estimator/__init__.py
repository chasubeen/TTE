"""Slim canonical estimator package — MLP manifold projector.

The canonical pipeline uses only `SubspaceEstimator` and `MLPProjector` from
this package. Deferred extensions (refit, calibrate, spatial filter, etc.)
live in `experimental/mlp_refit/`.
"""
from src.estimator.mlp_projector import MLPProjector, SubspaceEstimator

__all__ = ["MLPProjector", "SubspaceEstimator"]
