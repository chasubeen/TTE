"""selector — expansion-target selection (Req-B / Memory-independent Gate).

Owns the MLP layer that decides which test-stream patches are confident-normal
and thus eligible for memory expansion:
  - `MLPProjector` / `SubspaceEstimator` : the M₀-trained frozen MLP autoencoder
  - `Selector`                           : the e(p) < τ_low gate over that MLP
"""
from src.selector.mlp_gate import MLPProjector, SubspaceEstimator
from src.selector.gate import Selector

__all__ = ["MLPProjector", "SubspaceEstimator", "Selector"]
