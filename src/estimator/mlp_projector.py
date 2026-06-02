"""Backward-compatibility shim — moved to `src.selector.mlp_gate`.

The MLP gate now lives in the `selector` package (expansion-target selection).
Existing imports `from src.estimator.mlp_projector import ...` keep working.
"""
from src.selector.mlp_gate import MLPProjector, SubspaceEstimator

__all__ = ["MLPProjector", "SubspaceEstimator"]
